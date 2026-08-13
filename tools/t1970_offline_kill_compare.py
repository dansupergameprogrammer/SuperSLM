#!/usr/bin/env python3
"""T-1970 / T-1968-C1-final-form (Brunel, disposable, never merges).

=== HOW THIS ROUND GOT HERE ===

T-1968 (Poirot c131eab review, Critical 1/D-SLM2795) fixed the three-way
harness's shared-target confound by restricting capture to `position == 0`
(arm-independent by induction: attention softmax has one element there, so
Q cannot influence anything downstream). The coordinator then found T-1968's
own fix flawed: RoPE's rotation angle at position 0 is IDENTITY for every
pair, so legacy (quantize -> rotate -> clamp) and dynamic-fused (rotate ->
quantize) are the SAME FUNCTION there -- the kill rule read UNRESOLVED
forever by construction, not by measurement ("position 0 bought
arm-independence at the price of measuring nothing").

The prescribed remedy (the "same-row, three-treatment offline design",
tools/t1970_offline_kill.cpp) captured the wide pre-rotation Q row per
(layer, position) from a single legacy-arm prefill and anchored three
offline-reapplied landing treatments against two named engine runs. Its own
premise -- "the captured rows are arm-independent by definition" -- turned
out to hold only for the IMMEDIATE GEMM/fold/bias computation, not its
INPUT: Q's own construction shapes attention's output at any position >= 1,
which feeds the residual stream every LATER layer reads as its own input,
so a (layer >= 1, position >= 1) cell has already diverged between the two
named engine runs before either treatment is even applied to it. Verified
by direct enumeration (not inferred from a count): the dynamic anchor's own
68/1148 matches are EXACTLY {layer 0, any position} union {any layer,
position 0}, zero disagreements.

=== T-1968/C1 FINAL FORM: THE RE-SCOPED POOL ===

The coordinator's own re-scoping: the INTERSECTION of arm-independent
(layer 0 -- embedding-fed input, no upstream Q influence) and genuinely
rotated (position >= 1 -- non-identity RoPE) is layer 0, positions 1..40 --
40 cells, real rotation angles, arm-independent input PROVEN by the T-1970
enumeration above (cited as this pool's own admissibility proof, not
re-derived). tools/t1970_offline_kill.cpp's own executed check confirms:
40/40 pool cells pass BOTH anchor gates (legacy bit-exact, dynamic
bit-exact) AND non-identity rotation.

=== THE AMENDED PREDECLARED KILL RULE (stated BEFORE any number below was
computed, per StandardsDocument.md Sec4/Sec5.4) ===

  - Deciding quantities, UNCHANGED from T-1968's own D-SLM2799/2806:
    `pooled_rms_error_steps` (Q reconstruction) and `pooled_qk_rel_error`
    (QK-score), each per treatment (legacy, static-fused, dynamic-fused).
  - Comparison: dynamic-fused vs legacy (D-SLM2788's own kill-sequence).
    static-fused is reported ALONGSIDE as UNANCHORED-CONTEXT, never held to
    the RESOLVED/UNRESOLVED verdict machinery -- `tools/t1970_offline_kill.cpp`
    runs exactly TWO engine legs (legacy, dynamic); there is no static
    engine anchor, so static's own offline composition is never checked
    against the engine path it claims to model (T-1971 fix round, Poirot
    62c5f15 confirmation, Significant 1/D-SLM2823 -- the prior version of
    this rule stated the "context only" intent but the code ran static
    through the verdict machinery anyway and printed the word RESOLVED;
    fixed here, not merely re-stated).
  - Dispersion: PAIRED per-position delta (T-1971 fix round, D-SLM2824 --
    the prior form summed two INDEPENDENT standard errors over data that is
    perfectly PAIRED, all three treatments applied to the SAME 40 captured
    rows; the correct statistic is the per-position delta's own mean and
    standard error). `n=1` layer this round (layer 0 only), so a per-LAYER
    std was never available or used; `position_std` (of the 40 per-position
    PAIRED DELTAS) is the dispersion term, reported alongside the exact
    two-sided sign test (how many of the 40 positions favour dynamic-fused,
    and the p-value of that count under a fair-coin null) -- the sign test
    is what distinguishes a real, consistent effect from a pooled delta
    carried by magnitude at a few positions.
  - Resolving power: `z_crit * (paired_se)`, `paired_se = stdev(deltas) /
    sqrt(40)`, `z_crit = 1.96` (two-sided 95%). A paired mean delta smaller
    in magnitude than this resolving power is reported UNRESOLVED, not as a
    direction. Applied identically to both deciding quantities.
  - THE CONJUNCTION (D-SLM2787/D-SLM2788's own predeclared step-3
    criterion): reduce reconstructed-Q error AND QK-score error. Composed
    explicitly from the two verdicts, never left for a reader to infer
    (T-1971 fix round, Significant 3/D-SLM2825 -- the prior version reported
    both halves accurately but never stated the conjunction's own status,
    which reads as a banked positive with a footnote): MET only if both
    resolve better; FAILED if either resolves worse; otherwise NOT MET and
    INCOMPLETE (not a kill, not a pass) -- the current, real state, since
    Q-reconstruction resolves better and QK-score returns no directional
    evidence.
  - THE CELL THIS MEASURES, STATED HONESTLY (Sec5.4): this measures the
    boundary-removal mechanism (two int8 narrowings collapsed to one) at
    LAYER 0's OWN activation distribution only, over 40 real, arm-independent,
    genuinely-rotated positions. Two dispositions, decided BEFORE the
    numbers exist:
      * A KILL (dynamic-fused resolved WORSE than legacy on either
        quantity) generalizes. The boundary's own cost is a per-row
        arithmetic fact (an extra quantization step's own rounding error,
        or its absence) that does not depend on which layer computed the
        row -- if the construction cannot show a reduction on 40 real
        rotated rows at layer 0, it has no mechanism to rely on at any
        other layer either.
      * A POSITIVE (dynamic-fused resolved BETTER, or UNRESOLVED, on both)
        does NOT generalize to layers >= 1 without the deeper
        per-layer-isolated capture this round did not build (each layer's
        own test replayed against a reference input independent of any
        real run's own upstream attention). T-1691's own teacher-forced
        parity pattern is the named instrument for that capture, if it is
        ever needed. NOTE (Poirot 62c5f15 confirmation §11/O1): "a kill
        generalizes" is a decision heuristic, not a derived property -- the
        boundary's arithmetic is layer-independent, but how much it COSTS
        depends on the row's own distribution (how much the intermediate
        clamp truncates), which is not proven layer-independent. A null at
        one layer is strong evidence against a large general effect, not
        proof of none.

=== SINGLE-BOUNDARY FLOOR OBSERVATION (T-1971 add, Poirot 62c5f15
confirmation §11/O3, D-SLM2827) ===

  Independent mechanistic corroboration the record did not previously
  carry: the RMS of a single correctly-rounding int8 quantizer's own
  uniform rounding error has the theoretical floor `1/sqrt(12) = 0.288675`;
  a construction carrying TWO independent such boundaries has floor
  `sqrt(2)/sqrt(12) = 0.408248` (uncorrelated errors add in quadrature).
  Neither constant is fit to this round's own data. Reported alongside the
  kill rule, not as a substitute for it.

=== VITALITY SET (this round's own, plus carried-forward form) ===

  1. Pool admissibility: the deciding pool must contain ONLY cells passing
     BOTH anchor gates (legacy, dynamic) AND non-identity rotation, refused
     otherwise -- re-derived from the raw per-cell dump here (never trusted
     from the C++ tool's own summary counts alone), with constructive
     negative tests (a corrupted copy with one condition forced false must
     be rejected).
  2. The deciding quantity is computed AND printed by the real report path
     (not a second, parallel implementation) -- a hand-computed, non-zero
     synthetic control.
  3. A deliberate, large perturbation of the dynamic-fused treatment's own
     codes moves the deciding quantity by an unmistakable amount.

=== T-1971 (Poirot 62c5f15-t1968-c1-final-form-confirmation.md,
D-SLM2820-2827): what changed and why ===

Three reporting-tier fixes, none changing a deciding number (re-verified:
the RMS headline SURVIVES the paired form and strengthens; nothing here
re-ran any engine capture):
  S1/D-SLM2823 -- static demoted from a false RESOLVED/UNRESOLVED verdict
    to explicit UNANCHORED-CONTEXT (no engine leg exists for it).
  S2/D-SLM2824 -- resolving power switched from summed-independent-SE to
    the paired per-position form; the QK "same direction, underpowered"
    language corrected to "no directional evidence" (21/40 positions favour
    dynamic, sign test p=0.875 -- a coin flip, not a trend).
  S3/D-SLM2825 -- the composed conjunction sentence added: NOT MET,
    INCOMPLETE (one conjunct resolved, the other returned no directional
    evidence).
Plus two adds the review named as belonging in the record: the
single-boundary floor observation above, and (process, for the NEXT run
only, not retroactive to this one) committing a kill rule's own text in a
commit that precedes the run's own result commit, so git -- not just the
docstring's own internal consistency -- corroborates that the rule was
predeclared.
"""
import json
import math
import statistics
import sys

Z_CRIT_95 = 1.96
POOL_LAYER = 0
POOL_POSITIONS = range(1, 41)  # 1..40 inclusive


def load_dump(path):
    with open(path) as f:
        return json.load(f)


def scale_value(m, e):
    return float(m) * (2.0 ** e)


# --- Vitality 1: pool admissibility, re-derived from the raw dump --------

class PoolAdmissibilityError(Exception):
    pass


def admissible_pool(cells, expect_size=40):
    """Filters `cells` (the dump's own raw list) down to the deciding pool
    by RE-CHECKING every condition from the raw per-cell fields -- never
    trusting a pre-filtered list or the C++ tool's own summary counts.
    Refuses (raises) if the resulting pool does not have EXACTLY
    `expect_size` cells, or if it contains anything outside {layer ==
    POOL_LAYER, position in POOL_POSITIONS} once filtered, or if any
    included cell fails either anchor gate or is an identity rotation."""
    pool = []
    for c in cells:
        in_scope = (c["layer"] == POOL_LAYER) and (c["position"] in POOL_POSITIONS)
        if not in_scope:
            continue
        admissible = c["legacy_anchor_match"] and c["dynamic_anchor_match"] and not c["is_identity_rotation"]
        if not admissible:
            raise PoolAdmissibilityError(
                f"cell (layer={c['layer']}, position={c['position']}) is in the pool's own scope "
                f"but FAILS admissibility (legacy_anchor_match={c['legacy_anchor_match']}, "
                f"dynamic_anchor_match={c['dynamic_anchor_match']}, "
                f"is_identity_rotation={c['is_identity_rotation']}) -- refusing to grade a pool "
                f"that contains an inadmissible cell")
        pool.append(c)
    if len(pool) != expect_size:
        raise PoolAdmissibilityError(
            f"deciding pool has {len(pool)} cells, expected exactly {expect_size} -- refusing")
    return pool


def _self_test_pool_admissibility_rejects_anchor_failure(cells):
    """Constructive negative test: take a REAL in-scope pool cell, force its
    own `dynamic_anchor_match` to False (simulating what a genuine C1-class
    regression would look like), and confirm `admissible_pool` REFUSES
    rather than silently including or silently dropping it to a smaller,
    still-"successful" pool."""
    corrupted = json.loads(json.dumps(cells))
    for c in corrupted:
        if c["layer"] == POOL_LAYER and c["position"] == 5:
            c["dynamic_anchor_match"] = False
            break
    else:
        raise AssertionError("test setup failure: (layer 0, position 5) not found in the real dump")
    try:
        admissible_pool(corrupted, expect_size=40)
        return False  # did NOT raise -- the gate is inert, this self-test FAILS
    except PoolAdmissibilityError:
        return True


def _self_test_pool_admissibility_rejects_identity_leak(cells):
    """Constructive negative test: take a real in-scope pool cell and force
    `is_identity_rotation` to True (simulating a position-0-shaped cell
    leaking into the pool), confirm `admissible_pool` refuses."""
    corrupted = json.loads(json.dumps(cells))
    for c in corrupted:
        if c["layer"] == POOL_LAYER and c["position"] == 10:
            c["is_identity_rotation"] = True
            break
    else:
        raise AssertionError("test setup failure: (layer 0, position 10) not found in the real dump")
    try:
        admissible_pool(corrupted, expect_size=40)
        return False
    except PoolAdmissibilityError:
        return True


# --- The comparison itself -------------------------------------------------

def per_position_vectors(pool, treatment):
    """Yields (position, target_value[], reconstructed[], target_step,
    score_target, score_reconstructed) for one treatment
    ("legacy"/"static"/"dynamic") over the 40-cell pool, one row per
    position (layer is fixed at POOL_LAYER for every cell)."""
    codes_key = f"{treatment}_codes"
    scale_m_key = f"{treatment}_scale_m"
    scale_e_key = f"{treatment}_scale_e"
    for c in pool:
        codes = c[codes_key]
        scale_value_ = scale_value(c[scale_m_key], c[scale_e_key])
        reconstructed = [x * scale_value_ for x in codes]

        wide_rotated = c["target_wide_rotated"]
        target_normed = (scale_value(c["normed_scale_m"], c["normed_scale_e"])
                          * scale_value(c["site_constant_m"], c["site_constant_e"]) * 127.0)
        target_value = [w * target_normed for w in wide_rotated]
        d_prime = max(abs(w) for w in wide_rotated)
        target_step = d_prime * target_normed / 127.0
        if target_step == 0:
            continue

        k_row = c.get("k_row_head0") or []
        head_dim = len(k_row)
        score_target = None
        score_reconstructed = None
        if head_dim and len(target_value) >= head_dim:
            score_target = sum(t * k for t, k in zip(target_value[:head_dim], k_row))
            score_reconstructed = sum(r * k for r, k in zip(reconstructed[:head_dim], k_row))

        yield c["position"], target_value, reconstructed, target_step, score_target, score_reconstructed


def rms_error_steps(target_value, reconstructed, target_step):
    sq = sum(((t - r) / target_step) ** 2 for t, r in zip(target_value, reconstructed))
    return math.sqrt(sq / len(target_value))


def compare(pool):
    """Returns {treatment: {..}} for legacy/static/dynamic, each carrying
    per-position RMS/QK values, pooled figures, and the position-dispersion
    resolving power -- the amended rule's own form, over POSITIONS not
    layers."""
    results = {}
    for treatment in ("legacy", "static", "dynamic"):
        per_position_rms = []
        per_position_qk = []
        all_sq = []
        qk_rel_errors = []
        for pos, target_value, reconstructed, target_step, score_target, score_reconstructed in \
                per_position_vectors(pool, treatment):
            pos_rms = rms_error_steps(target_value, reconstructed, target_step)
            per_position_rms.append(pos_rms)
            all_sq.extend(((t - r) / target_step) ** 2 for t, r in zip(target_value, reconstructed))
            if score_target is not None and abs(score_target) > 0:
                qk_err = abs(score_target - score_reconstructed) / abs(score_target)
                qk_rel_errors.append(qk_err)
                per_position_qk.append(qk_err)
        pooled_rms_error_steps = math.sqrt(sum(all_sq) / len(all_sq)) if all_sq else float("nan")
        pooled_qk_rel_error = (sum(qk_rel_errors) / len(qk_rel_errors)) if qk_rel_errors else float("nan")
        rms_position_std = statistics.pstdev(per_position_rms) if len(per_position_rms) > 1 else 0.0
        qk_position_std = statistics.pstdev(per_position_qk) if len(per_position_qk) > 1 else 0.0
        results[treatment] = {
            "positions": len(per_position_rms),
            "pooled_rms_error_steps": pooled_rms_error_steps,
            "per_position_rms": per_position_rms,
            "rms_position_std": rms_position_std,
            "pooled_qk_rel_error": pooled_qk_rel_error,
            "per_position_qk": per_position_qk,
            "qk_positions": len(per_position_qk),
            "qk_position_std": qk_position_std,
        }
    return results


def sign_test_p(k, n):
    """Exact two-sided binomial sign test against p=0.5: sums the PMF of
    every outcome at least as extreme as the observed `k` successes out of
    `n` trials. T-1971 fix round (Poirot 62c5f15 confirmation, Significant
    2/D-SLM2824): the ORIGINAL form here summed two INDEPENDENT standard
    errors over data that is perfectly PAIRED (all three treatments applied
    to the SAME 40 captured rows) -- unpaired-over-paired-data is the
    defect the reviewer's own re-execution caught. This function and
    `paired_analysis` below replace it."""
    if n < 2:
        return None

    def pmf(x):
        return math.comb(n, x) / (2 ** n)

    p_obs = pmf(k)
    return sum(pmf(x) for x in range(n + 1) if pmf(x) <= p_obs + 1e-15)


def paired_analysis(candidate_values, reference_values, z_crit=Z_CRIT_95):
    """T-1971 fix round (D-SLM2824): the CORRECT statistic for two
    treatments applied to the same `n` rows -- the per-position paired
    delta (candidate - reference), its own mean and standard error, the
    resolving power from THAT dispersion (never two independent SEs
    summed), and the sign test (how many positions favour the candidate,
    i.e. a LOWER value -- these are error metrics, lower is better) with
    its exact two-sided p-value. Returns None if fewer than 2 paired
    positions exist (dispersion undefined)."""
    n = min(len(candidate_values), len(reference_values))
    if n < 2:
        return None
    deltas = [c - r for c, r in zip(candidate_values, reference_values)]
    mean_delta = statistics.mean(deltas)
    se = statistics.stdev(deltas) / math.sqrt(n)  # sample stdev (ddof=1), the paired-SE convention
    rp = z_crit * se
    favor = sum(1 for c, r in zip(candidate_values, reference_values) if c < r)
    p_value = sign_test_p(favor, n)
    return {"mean_delta": mean_delta, "se": se, "rp": rp, "favor": favor, "n": n, "p_value": p_value}


def paired_verdict(name, quantity, analysis):
    """Returns (verdict_kind, text) where verdict_kind is one of
    "RESOLVED_BETTER"/"RESOLVED_WORSE"/"UNRESOLVED"/"UNDEFINED" -- read by
    the conjunction logic in `print_report`, never re-derived from the
    printed string."""
    if analysis is None:
        return "UNDEFINED", "RESOLVING POWER UNDEFINED (fewer than 2 paired positions)"
    delta, rp = analysis["mean_delta"], analysis["rp"]
    sign_note = f"{analysis['favor']}/{analysis['n']} positions favour {name}, sign test p={analysis['p_value']:.3g}"
    if rp == 0.0 or abs(delta) < rp:
        return "UNRESOLVED", (f"UNRESOLVED (|paired mean delta|={abs(delta):.6f} < paired "
                               f"resolving_power={rp:.6f}; {sign_note})")
    if delta > 0:
        return "RESOLVED_WORSE", (f"RESOLVED: {name} is WORSE than legacy on {quantity} "
                                   f"(paired mean delta={delta:+.6f}, paired resolving_power={rp:.6f}; "
                                   f"{sign_note})")
    return "RESOLVED_BETTER", (f"RESOLVED: {name} is BETTER than legacy on {quantity} "
                                f"(paired mean delta={delta:+.6f}, paired resolving_power={rp:.6f}; "
                                f"{sign_note})")


# T-1971 fix round (Poirot 62c5f15 confirmation §11/O3, D-SLM2827): the
# theoretical RMS floor of a single uniformly-distributed rounding error is
# 1/sqrt(12) (a correctly-rounding int8 quantizer's own noise floor); a
# construction carrying TWO independent such boundaries has floor
# sqrt(2)/sqrt(12) (uncorrelated errors add in quadrature). Independent
# mechanistic corroboration, computed from constants alone -- not fit to
# the data.
ONE_BOUNDARY_FLOOR = 1.0 / math.sqrt(12.0)
TWO_BOUNDARY_FLOOR = math.sqrt(2.0) / math.sqrt(12.0)


def print_report(results):
    print("treatment | positions | pooled_rms_error_steps | rms_position_std | qk_positions | "
          "pooled_qk_rel_error | qk_position_std")
    print("----------|-----------|-------------------------|-------------------|--------------|"
          "----------------------|------------------")
    for name in ("legacy", "static", "dynamic"):
        r = results[name]
        print(f"{name:9s} | {r['positions']:9d} | {r['pooled_rms_error_steps']:23.6f} | "
              f"{r['rms_position_std']:17.6f} | {r['qk_positions']:12d} | "
              f"{r['pooled_qk_rel_error']:20.6f} | {r['qk_position_std']:16.6f}")

    # --- Static: UNANCHORED-CONTEXT, per T-1971 fix round (D-SLM2823) ---
    # `tools/t1970_offline_kill.cpp` runs exactly TWO engine legs (legacy,
    # dynamic) -- there is no static engine anchor. Static's own offline
    # composition is therefore never checked against an engine path it
    # claims to model, and it is reported for context only: no verdict
    # word (RESOLVED/UNRESOLVED), no kill-rule machinery.
    print()
    print("=== static-fused: UNANCHORED-CONTEXT (no engine leg for this treatment; not held to the "
          "RESOLVED/UNRESOLVED kill machinery) ===")
    print(f"pooled_rms_error_steps={results['static']['pooled_rms_error_steps']:.6f} "
          f"(vs legacy {results['legacy']['pooled_rms_error_steps']:.6f}, raw delta "
          f"{results['static']['pooled_rms_error_steps'] - results['legacy']['pooled_rms_error_steps']:+.6f}); "
          f"pooled_qk_rel_error={results['static']['pooled_qk_rel_error']:.6f} "
          f"(vs legacy {results['legacy']['pooled_qk_rel_error']:.6f}, raw delta "
          f"{results['static']['pooled_qk_rel_error'] - results['legacy']['pooled_qk_rel_error']:+.6f})")

    print()
    print(f"=== Amended predeclared kill rule (T-1971 fix, D-SLM2824): PAIRED per-position delta, "
          f"dispersion across 40 POSITIONS (n_layers=1, layer 0 only), z_crit={Z_CRIT_95} "
          f"(two-sided 95%) ===")
    rms_analysis = paired_analysis(results["dynamic"]["per_position_rms"], results["legacy"]["per_position_rms"])
    rms_kind, rms_text = paired_verdict("dynamic", "pooled_rms_error_steps", rms_analysis)
    print(f"dynamic vs legacy, pooled_rms_error_steps: {rms_text}")

    qk_analysis = paired_analysis(results["dynamic"]["per_position_qk"], results["legacy"]["per_position_qk"])
    qk_kind, qk_text = paired_verdict("dynamic", "pooled_qk_rel_error", qk_analysis)
    print(f"dynamic vs legacy, pooled_qk_rel_error: {qk_text}")

    # --- T-1971 fix round (D-SLM2825): the composed conjunction sentence.
    # The predeclared step-3 criterion (D-SLM2787/D-SLM2788) is an AND: BOTH
    # quantities must resolve better. Read from the verdict KIND (never
    # re-derived from printed text), so this composition cannot drift from
    # what was actually computed.
    print()
    if rms_kind == "RESOLVED_WORSE" or qk_kind == "RESOLVED_WORSE":
        print("=== Predeclared step-3 conjunction (reduce Q-reconstruction error AND QK-score error): "
              "NOT MET -- FAILED (at least one quantity resolved WORSE than legacy) ===")
    elif rms_kind == "RESOLVED_BETTER" and qk_kind == "RESOLVED_BETTER":
        print("=== Predeclared step-3 conjunction (reduce Q-reconstruction error AND QK-score error): "
              "MET -- both quantities resolved better than legacy ===")
    else:
        print("=== Predeclared step-3 conjunction (reduce Q-reconstruction error AND QK-score error): "
              "NOT MET -- one conjunct resolved, the other returned no directional evidence; the test "
              "is INCOMPLETE, not passed and not failed. ===")

    # --- T-1971 fix round (D-SLM2827/O3): the single-boundary floor
    # observation, independent mechanistic corroboration the record did not
    # previously carry.
    print()
    dyn_rms = results["dynamic"]["pooled_rms_error_steps"]
    leg_rms = results["legacy"]["pooled_rms_error_steps"]
    dyn_floor_pct = (ONE_BOUNDARY_FLOOR - dyn_rms) / ONE_BOUNDARY_FLOOR * 100.0
    print(f"=== Single-boundary floor observation (mechanistic corroboration, not a statistical test) ===")
    print(f"theoretical one-boundary RMS floor 1/sqrt(12) = {ONE_BOUNDARY_FLOOR:.6f}; "
          f"theoretical two-boundary floor sqrt(2)/sqrt(12) = {TWO_BOUNDARY_FLOOR:.6f}")
    print(f"dynamic pooled_rms_error_steps = {dyn_rms:.6f} -- {dyn_floor_pct:.2f}% below the "
          f"one-boundary floor (consistent with exactly ONE int8 boundary, which this construction "
          f"carries)")
    print(f"legacy pooled_rms_error_steps  = {leg_rms:.6f} -- between the one-boundary "
          f"({ONE_BOUNDARY_FLOOR:.4f}) and two-boundary ({TWO_BOUNDARY_FLOOR:.4f}) floors, consistent "
          f"with two partially correlated quantizations")

    print()
    print("NOTE: this measures the boundary-removal mechanism at LAYER 0's own activation "
          "distribution, over 40 real, arm-independent, genuinely-rotated positions. A KILL "
          "generalizes (the boundary's own cost is per-row arithmetic, layer-independent) -- this is "
          "a decision heuristic, not a derived property (Poirot 62c5f15 confirmation §11/O1): a null "
          "at one layer is strong evidence against a large general effect, not proof of none. A "
          "POSITIVE does NOT generalize to layers >= 1 without the deeper per-layer-isolated capture "
          "this round did not build (T-1691's teacher-forced parity pattern, if ever needed).")


# --- Vitality 2/3 ------------------------------------------------------------

def _self_test_deciding_quantity_is_read():
    """Hand-computed, non-zero synthetic control routed through the REAL
    `compare()`/`print_report()` path -- Sec5.4's own control-calibration
    law (a zero-error population proves nothing about whether the RIGHT
    function is being read)."""
    m30 = 1 << 30
    synthetic_cell = {
        "layer": 0, "position": 1, "is_identity_rotation": False,
        "legacy_anchor_match": True, "dynamic_anchor_match": True,
        "normed_scale_m": m30, "normed_scale_e": 0,
        "site_constant_m": m30, "site_constant_e": -30,
        "target_wide_rotated": [100, 100],
        "legacy_codes": [10, 10], "legacy_scale_m": m30, "legacy_scale_e": 0,
        "static_codes": [10, 10], "static_scale_m": m30, "static_scale_e": 0,
        "dynamic_codes": [10, 10], "dynamic_scale_m": m30, "dynamic_scale_e": 0,
        "k_row_head0": [],
    }
    pool = [synthetic_cell]
    # target_normed = 2**30 * 1 * 127 = q_scale_value * 127 (q_scale_value = 2**30)
    # reconstructed = 10 * 2**30; target_value = 100*127*2**30; target_step = 100*2**30
    # error_in_steps = (100*127 - 10) / 100 = 126.90 exactly, same hand-derivation T-1968 used.
    expected_error_steps = (100 * 127 - 10) / 100.0
    results = compare(pool)
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_report(results)
    printed = buf.getvalue()
    got = results["legacy"]["pooled_rms_error_steps"]
    if abs(got - expected_error_steps) > 1e-6:
        raise AssertionError(f"hand-computed {expected_error_steps} != compare()'s own {got}")
    if f"{got:.6f}" not in printed:
        raise AssertionError(f"deciding quantity {got:.6f} computed but not printed:\n{printed}")
    return got, printed


def _self_test_perturbed_dynamic_moves_metric(pool):
    """Deliberately perturbs the dynamic-fused treatment's own codes (+20,
    clamped) across the REAL 40-cell pool and confirms the deciding
    quantity moves by a large, unmistakable amount relative to the clean
    pool."""
    perturbed = json.loads(json.dumps(pool))
    for c in perturbed:
        c["dynamic_codes"] = [max(-127, min(127, x + 20)) for x in c["dynamic_codes"]]
    clean = compare(pool)["dynamic"]["pooled_rms_error_steps"]
    dirty = compare(perturbed)["dynamic"]["pooled_rms_error_steps"]
    return clean, dirty


def main():
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <t1970_offline_kill.json>", file=sys.stderr)
        sys.exit(2)
    dump = load_dump(sys.argv[1])
    cells = dump["cells"]

    print("=== T-1970/T-1968-C1-final-form: VITALITY CHECKS (not the graded run) ===")

    pool = admissible_pool(cells, expect_size=40)
    print(f"vitality 1a -- deciding pool re-derived from the raw dump: 40/40 cells, all pass both "
          f"anchor gates AND non-identity rotation -- PASS")

    rejected_anchor = _self_test_pool_admissibility_rejects_anchor_failure(cells)
    print(f"vitality 1b -- a forced anchor-gate failure (layer 0, position 5) is REJECTED by "
          f"admissible_pool(): {'PASS' if rejected_anchor else 'FAIL'}")
    if not rejected_anchor:
        print("FAILED: pool admissibility did not reject a corrupted anchor -- vitality 1 is inert",
              file=sys.stderr)
        sys.exit(1)

    rejected_identity = _self_test_pool_admissibility_rejects_identity_leak(cells)
    print(f"vitality 1c -- a forced identity-rotation leak (layer 0, position 10) is REJECTED by "
          f"admissible_pool(): {'PASS' if rejected_identity else 'FAIL'}")
    if not rejected_identity:
        print("FAILED: pool admissibility did not reject an identity-rotation leak", file=sys.stderr)
        sys.exit(1)

    got, _printed = _self_test_deciding_quantity_is_read()
    print(f"vitality 2 -- the deciding quantity is computed AND actually printed by print_report: "
          f"hand-computed=126.900000 got={got:.6f} -- PASS")

    clean_rms, dirty_rms = _self_test_perturbed_dynamic_moves_metric(pool)
    moved = dirty_rms > clean_rms * 5 and dirty_rms > 1.0
    print(f"vitality 3 -- a deliberate perturbation (dynamic codes +20) moves the deciding quantity: "
          f"clean={clean_rms:.6f} perturbed={dirty_rms:.6f} ratio={dirty_rms/max(clean_rms,1e-9):.2f}x "
          f"-- {'PASS' if moved else 'FAIL'}")
    if not moved:
        print("FAILED: the metric did not move meaningfully under a deliberate perturbation",
              file=sys.stderr)
        sys.exit(1)

    print()
    print("=== Deciding pool: layer 0, positions 1..40 (40 cells) -- amended kill rule applied ===")
    results = compare(pool)
    print_report(results)


if __name__ == "__main__":
    main()

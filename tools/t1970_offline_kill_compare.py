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
    static-fused is reported ALONGSIDE as context, not held to the same
    RESOLVED/UNRESOLVED verdict machinery (it was never the candidate this
    kill sequence tests -- D-SLM2788 named dynamic-fused as the
    single-variable construction).
  - Dispersion: across the 40 POSITIONS, not per-layer -- n=1 layer this
    round (layer 0 only), so a per-LAYER std is undefined by construction;
    `position_std` (the standard deviation of the 40 per-position RMS/
    QK-error values) is the dispersion term instead, stated here explicitly
    so no reader mistakes it for T-1968's own per-layer figure.
  - Resolving power: `z_crit * (position_std / sqrt(40))`, `z_crit = 1.96`
    (two-sided 95%, this campaign's own established convention). A pooled
    delta smaller in magnitude than the combined (dynamic + legacy)
    resolving power is reported UNRESOLVED, not as a direction. Applied
    identically to both deciding quantities.
  - THE CELL THIS MEASURES, STATED HONESTLY (Sec5.4): this measures the
    boundary-removal mechanism (two int8 narrowings collapsed to one) at
    LAYER 0's OWN activation distribution only, over 40 real, arm-independent,
    genuinely-rotated positions. Two dispositions, decided BEFORE the
    numbers exist:
      * A KILL (dynamic-fused resolved WORSE than legacy) generalizes. The
        boundary's own cost is a per-row arithmetic fact (an extra
        quantization step's own rounding error, or its absence) that does
        not depend on which layer computed the row -- if the construction
        cannot show a reduction on 40 real rotated rows at layer 0, it has
        no mechanism to rely on at any other layer either.
      * A POSITIVE (dynamic-fused resolved BETTER, or UNRESOLVED) does NOT
        generalize to layers >= 1 without the deeper per-layer-isolated
        capture this round did not build (each layer's own test replayed
        against a reference input independent of any real run's own
        upstream attention). T-1691's own teacher-forced parity pattern is
        the named instrument for that capture, if it is ever needed.

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
            "qk_positions": len(per_position_qk),
            "qk_position_std": qk_position_std,
        }
    return results


def resolving_power(std, n, z_crit=Z_CRIT_95):
    if n < 2:
        return None
    return z_crit * (std / math.sqrt(n))


def verdict(name, delta, rp_a, rp_b):
    combined_rp = (rp_a or 0.0) + (rp_b or 0.0) if rp_a is not None and rp_b is not None else None
    if combined_rp is None or combined_rp == 0.0:
        return "RESOLVING POWER UNDEFINED (fewer than 2 positions on one side)"
    if abs(delta) < combined_rp:
        return f"UNRESOLVED (|delta|={abs(delta):.6f} < resolving_power={combined_rp:.6f})"
    direction = "WORSE than" if delta > 0 else "BETTER than"
    return f"RESOLVED: {name} is {direction} legacy (delta={delta:+.6f}, resolving_power={combined_rp:.6f})"


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

    print()
    print(f"=== Amended predeclared kill rule, dispersion across 40 POSITIONS (n_layers=1, layer 0 "
          f"only), z_crit={Z_CRIT_95} (two-sided 95%) ===")
    for name in ("static", "dynamic"):
        rms_rp = resolving_power(results[name]["rms_position_std"], results[name]["positions"])
        rms_rp_ref = resolving_power(results["legacy"]["rms_position_std"], results["legacy"]["positions"])
        delta_rms = results[name]["pooled_rms_error_steps"] - results["legacy"]["pooled_rms_error_steps"]
        print(f"{name} vs legacy, pooled_rms_error_steps: {verdict(name, delta_rms, rms_rp, rms_rp_ref)}")

        qk_rp = resolving_power(results[name]["qk_position_std"], results[name]["qk_positions"])
        qk_rp_ref = resolving_power(results["legacy"]["qk_position_std"], results["legacy"]["qk_positions"])
        delta_qk = results[name]["pooled_qk_rel_error"] - results["legacy"]["pooled_qk_rel_error"]
        print(f"{name} vs legacy, pooled_qk_rel_error: {verdict(name, delta_qk, qk_rp, qk_rp_ref)}")
    print()
    print("NOTE: this measures the boundary-removal mechanism at LAYER 0's own activation "
          "distribution, over 40 real, arm-independent, genuinely-rotated positions. A KILL "
          "generalizes (the boundary's own cost is per-row arithmetic, layer-independent); a "
          "POSITIVE does NOT generalize to layers >= 1 without the deeper per-layer-isolated "
          "capture this round did not build (T-1691's teacher-forced parity pattern, if ever needed).")


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

#!/usr/bin/env python3
"""T-1966/T-1968 (Brunel, disposable, never merges; D-SLM2787/D-SLM2788,
fix round per Claude/Poirot/c131eab-t1966-dynamic-fusedq-harness-review.md).
The three-way numeric comparison harness's own comparator: reads the three
per-arm JSON captures tools/t1966_arm_capture.exe produces (legacy,
static-fused, dynamic-fused), reconstructs each arm's own landed Q at
POSITION 0 (T-1968's own Critical 1 fix -- see below) in a common,
unit-consistent real-valued space, and reports reconstructed-Q error and
QK-score error against the dynamic-fused capture's own wide, pre-narrowing
target -- per layer and pooled.

THIS SCRIPT IS THE HARNESS ITSELF. It is not run as the graded three-way
comparison in this ticket (the ticket's own STOP instruction) -- only
against the single real captured token this session produced, to prove the
harness's own machinery is sound before any real capture trusts it.

=== T-1968 FIX ROUND: WHY POSITION 0, AND WHY THAT MAKES THE SHARED TARGET
VALID AGAIN ===

T-1966's own first draft captured the LAST step of a full 16-token greedy
decode. The three arms' own decodes diverge autoregressively (different
generated tokens from the 2nd/15th token onward), so "the last step" was a
DIFFERENT token for each arm, embedding a DIFFERENT residual stream into
every layer -- grading all three against the dynamic-fused arm's own last
capture measured trajectory divergence, not quantization error (the
review's own executed correlations: 0.9997 self-compared, 0.8832/0.6485
for the other two -- Claude/Poirot/c131eab-t1966-dynamic-fusedq-harness-review.md
Critical 1, D-SLM2795).

FIX: the C++ side (`forward_sites.cpp`) now captures ONLY at
`position == 0`, never overwritten afterward. At position 0 the attention
softmax has exactly one element (width == 1), so it evaluates to 1.0
regardless of what Q's own value was -- Q cannot influence the attention
OUTPUT at this position, and nothing else downstream reads Q, so EVERY
layer's own residual stream at position 0 is identical across all three
arms, by induction from layer 0 (whose own input is the prompt's first
token embedding, arm-independent by construction). This is not assumed:
`assert_arm_identity_at_position0()` below reads `normed_scale`/
`k_row_head0` directly from the three captured dumps and REFUSES (raises)
if they are not bit-identical at every layer, BEFORE any score is computed
-- executed once per comparator run, not a one-time check. Once this
identity is verified, the ORIGINAL architecture (grading all three arms
against the dynamic-fused arm's own captured target) is valid again,
because the confound (residual-stream divergence) is now impossible by
construction, not merely assumed absent.

=== THE PREDECLARED KILL RULE (Significant 3/D-SLM2799) ===

Stated here, BEFORE any graded run, per the review's own required form:
minimum effect size, per-layer-vs-pooled shape, and resolving power stated
WITH the result, not after it is liked (StandardsDocument.md Sec4/Sec5.4).

  - Deciding quantities: `pooled_rms_error_steps` (Q reconstruction) and
    `pooled_qk_rel_error` (QK-score), each per arm.
  - Comparison: dynamic-fused vs legacy (D-SLM2788's own kill-sequence:
    "require reduced reconstructed-Q and QK-score error vs legacy").
  - Per-layer AND pooled form, both reported: the pooled figure is the RMS
    over every captured element across every layer; per-layer dispersion
    (std of the per-layer RMS values) is reported alongside it, never
    silently dropped.
  - Minimum effect size / resolving power: `resolving_power = z_crit *
    (per_layer_std / sqrt(n_layers))`, `z_crit = 1.96` (two-sided 95%,
    matching this campaign's own established convention -- e.g. T-1946/
    T-1960's own signed_ratio tables). A pooled delta smaller in magnitude
    than the resolving power is reported UNRESOLVED, not as a direction.
  - This round's own achieved resolving power is computed over 28 LAYERS
    of ONE token (this ticket's own captured sample), explicitly NOT the
    32-prompt population D-SLM2794 names for the graded run -- the real
    run's own resolving power will differ (likely tighter, from more
    samples) and must be recomputed then, not inherited from here.
"""
import json
import math
import statistics
import sys

Z_CRIT_95 = 1.96


def load_arm(path):
    with open(path) as f:
        return json.load(f)


def scale_value(m, e):
    return float(m) * (2.0 ** e)


# --- Critical 1 fix: the refusing arm-identity gate ----------------------

class ArmIdentityMismatch(Exception):
    pass


def assert_arm_identity_at_position0(arms):
    """T-1968 Critical 1 (D-SLM2795): before any score is computed, verify
    -- by reading the three dumps' own captured data, not by citing the
    mathematical argument alone -- that `normed_scale` and `k_row_head0`
    are bit-identical across all three arms at every layer both captured.
    Raises ArmIdentityMismatch (a REFUSING gate, not a warning) on any
    disagreement. `arms` is a dict of {arm_name: dump_dict}.
    """
    names = list(arms.keys())
    by_layer = {name: {L["layer"]: L for L in arms[name]["layers"]} for name in names}
    common_layers = set.intersection(*(set(by_layer[n].keys()) for n in names))
    if not common_layers:
        raise ArmIdentityMismatch("no layer is present in all three dumps -- cannot verify identity")

    mismatches = []
    for l in sorted(common_layers):
        recs = {n: by_layer[n][l] for n in names}
        # Only compare layers where the reference (first) arm actually
        # captured data -- a layer with an empty codes/k_row list was never
        # reached with capture enabled (not this fix's own concern) and is
        # left to per_layer_vectors' own missing-target handling.
        if not recs[names[0]]["codes"]:
            continue
        ns = {n: (recs[n]["normed_scale_m"], recs[n]["normed_scale_e"]) for n in names}
        kr = {n: recs[n]["k_row_head0"] for n in names}
        if len(set(ns.values())) > 1:
            mismatches.append(f"layer {l}: normed_scale differs: {ns}")
        if len({tuple(v) for v in kr.values()}) > 1:
            mismatches.append(f"layer {l}: k_row_head0 differs")
    if mismatches:
        raise ArmIdentityMismatch(
            f"{len(mismatches)} layer(s) show arm-dependent pre-quantization context -- the "
            f"shared-target comparison is INVALID for this capture:\n" + "\n".join(mismatches))
    return len(common_layers)


# --- Critical 2 fix: engine-sourced arm identity, not a caller's claim ---

def assert_dump_arm_identity(dump, expected_arm_mode):
    """T-1968 Critical 2 (D-SLM2796): the dump's own `arm_mode` field is the
    ENGINE's own readback (forward_sites.cpp's own `option_g_fused_q_mode`,
    captured per layer at position 0 by t1966_arm_capture.cpp, which
    itself already refused to write a dump if the engine's own answer
    disagreed with what the calling process asked for -- see that tool's
    own source). This function re-checks it here too, so the comparator
    itself does not trust a dump's `arm`/`arm_mode` fields without
    cross-checking them against each other and against every captured
    layer -- closing the reviewer's own relabelling attack at the point of
    USE, not only at the point of CAPTURE.
    """
    if "arm_mode" not in dump:
        raise ArmIdentityMismatch(f"dump has no engine-sourced arm_mode field at all -- "
                                   f"refusing to trust its arm=\"{dump.get('arm')}\" claim")
    if dump["arm_mode"] != expected_arm_mode:
        raise ArmIdentityMismatch(
            f"dump's own engine-sourced arm_mode={dump['arm_mode']} does not match the "
            f"expected {expected_arm_mode} for arm=\"{dump.get('arm')}\"")
    for layer_rec in dump["layers"]:
        if not layer_rec["codes"]:
            continue
        # t1966_arm_capture.cpp does not currently write a per-layer
        # arm_mode field into the JSON (only the top-level, already
        # cross-checked against every layer at capture time) -- this loop
        # is a placeholder for that stronger check should the capture
        # format grow one; today it is a no-op beyond the top-level check
        # above, stated rather than silently assumed complete.
        pass


# --- The comparison itself ------------------------------------------------

def per_layer_vectors(arm_data, target_data):
    """Yields (layer, target_value[], reconstructed_value[], target_step,
    score_target, score_reconstructed). REFUSES (raises) rather than
    silently skipping when an arm's own layer HAS data but the target
    does not, or vice versa at a layer either claims to have captured --
    per the review's own remedy ("the harness must additionally refuse,
    rather than skip, a layer whose target is missing")."""
    target_by_layer = {L["layer"]: L for L in target_data["layers"]}
    for layer_rec in arm_data["layers"]:
        l = layer_rec["layer"]
        codes = layer_rec["codes"]
        if not codes:
            continue  # this arm's own capture has nothing for this layer (position 0 was never
                       # reached, or anchor-capture was off) -- skip, not an error: a genuinely
                       # absent capture is different from a present-but-mismatched one.
        target_rec = target_by_layer.get(l)
        if target_rec is None or not target_rec["wide_rotated"]:
            raise ValueError(f"layer {l}: this arm captured Q data but the target dump has no "
                              f"wide_rotated for this layer -- refusing to pool over a partial "
                              f"population (Poirot c131eab review Critical 1's own remedy)")
        wide_rotated = target_rec["wide_rotated"]
        if len(wide_rotated) != len(codes):
            raise ValueError(f"layer {l}: target has {len(wide_rotated)} elements, "
                              f"arm has {len(codes)} -- geometry mismatch, cannot compare")

        normed_value = scale_value(layer_rec["q_scale_m"], layer_rec["q_scale_e"])
        reconstructed = [c * normed_value for c in codes]

        # target_normed = normed_scale_value * site_constant_value * 127 --
        # see the module docstring's own history for the two arithmetic
        # corrections this formula required (D-SLM2793); unchanged by this
        # fix round.
        target_normed = (scale_value(target_rec["normed_scale_m"], target_rec["normed_scale_e"])
                          * scale_value(target_rec["site_constant_m"], target_rec["site_constant_e"])
                          * 127.0)
        target_value = [w * target_normed for w in wide_rotated]
        d_prime = max(abs(w) for w in wide_rotated)
        target_step = d_prime * target_normed / 127.0
        if target_step == 0:
            continue

        k_row = layer_rec.get("k_row_head0") or target_rec.get("k_row_head0")
        head_dim = len(k_row) if k_row else 0
        score_target = None
        score_reconstructed = None
        if head_dim and len(target_value) >= head_dim:
            score_target = sum(t * k for t, k in zip(target_value[:head_dim], k_row))
            score_reconstructed = sum(r * k for r, k in zip(reconstructed[:head_dim], k_row))

        yield l, target_value, reconstructed, target_step, score_target, score_reconstructed


def rms_error_steps(target_value, reconstructed, target_step):
    sq = sum(((t - r) / target_step) ** 2 for t, r in zip(target_value, reconstructed))
    return math.sqrt(sq / len(target_value))


def compare(arms, verify_identity=True):
    """`arms` is {name: dump_dict}, must contain "dynamic-fused" (the
    target). Set `verify_identity=False` only for a synthetic self-test
    dump that has no k_row_head0/normed_scale variation to check (the
    real-data path always verifies)."""
    dynamic_fused = arms["dynamic-fused"]
    if verify_identity:
        assert_arm_identity_at_position0(arms)

    results = {}
    for name, arm_data in arms.items():
        per_layer_rms = []
        per_layer_qk = []
        all_sq = []
        qk_rel_errors = []
        for l, target_value, reconstructed, target_step, score_target, score_reconstructed in \
                per_layer_vectors(arm_data, dynamic_fused):
            layer_rms = rms_error_steps(target_value, reconstructed, target_step)
            per_layer_rms.append(layer_rms)
            all_sq.extend(((t - r) / target_step) ** 2 for t, r in zip(target_value, reconstructed))
            if score_target is not None and abs(score_target) > 0:
                qk_err = abs(score_target - score_reconstructed) / abs(score_target)
                qk_rel_errors.append(qk_err)
                per_layer_qk.append(qk_err)
        pooled_rms_error_steps = math.sqrt(sum(all_sq) / len(all_sq)) if all_sq else float("nan")
        pooled_qk_rel_error = (sum(qk_rel_errors) / len(qk_rel_errors)) if qk_rel_errors else float("nan")
        layer_std = statistics.pstdev(per_layer_rms) if len(per_layer_rms) > 1 else 0.0
        results[name] = {
            "layers": len(per_layer_rms),
            "pooled_rms_error_steps": pooled_rms_error_steps,
            "per_layer_rms": per_layer_rms,
            "per_layer_std": layer_std,
            "pooled_qk_rel_error": pooled_qk_rel_error,
            "qk_layers": len(qk_rel_errors),
        }
    return results


def resolving_power(result, z_crit=Z_CRIT_95):
    """Significant 3 fix (D-SLM2799): the achieved resolving power for this
    arm's own pooled_rms_error_steps, from its OWN per-layer dispersion --
    `z_crit * (std / sqrt(n))`, the standard-error-of-the-mean form this
    campaign already uses elsewhere (T-1946/T-1960's own signed_ratio
    tables). Returns None if fewer than 2 layers were captured (dispersion
    undefined)."""
    n = result["layers"]
    if n < 2:
        return None
    return z_crit * (result["per_layer_std"] / math.sqrt(n))


def print_report(results, reference_arm="legacy", z_crit=Z_CRIT_95):
    print("arm            | layers | pooled_rms_error_steps | per_layer_std | qk_layers | "
          "pooled_qk_rel_error")
    print("---------------|--------|-------------------------|---------------|-----------|"
          "---------------------")
    for name in ("legacy", "static-fused", "dynamic-fused"):
        r = results[name]
        # THE deciding quantity is read here and printed -- not a
        # different, silently-substituted number (D-SLM2750's own scar).
        print(f"{name:14s} | {r['layers']:6d} | {r['pooled_rms_error_steps']:23.6f} | "
              f"{r['per_layer_std']:13.6f} | {r['qk_layers']:9d} | {r['pooled_qk_rel_error']:.6f}")

    print()
    print(f"=== Predeclared kill rule, z_crit={z_crit} (two-sided 95%) ===")
    for name in ("static-fused", "dynamic-fused"):
        if name not in results or reference_arm not in results:
            continue
        delta = results[name]["pooled_rms_error_steps"] - results[reference_arm]["pooled_rms_error_steps"]
        rp = resolving_power(results[name], z_crit)
        rp_ref = resolving_power(results[reference_arm], z_crit)
        combined_rp = (rp or 0.0) + (rp_ref or 0.0) if rp is not None and rp_ref is not None else None
        if combined_rp is None or combined_rp == 0.0:
            verdict = "RESOLVING POWER UNDEFINED (fewer than 2 layers captured for one side)"
        elif abs(delta) < combined_rp:
            verdict = f"UNRESOLVED (|delta|={abs(delta):.6f} < resolving_power={combined_rp:.6f})"
        else:
            direction = "WORSE than" if delta > 0 else "BETTER than"
            verdict = (f"RESOLVED: {name} is {direction} {reference_arm} "
                       f"(delta={delta:+.6f}, resolving_power={combined_rp:.6f})")
        print(f"{name} vs {reference_arm}: {verdict}")
    print()
    print("NOTE: resolving power above is computed from 28 LAYERS OF ONE TOKEN (this ticket's own "
          "captured sample) -- explicitly NOT the 32-prompt population the graded run will use. "
          "Recompute against the real population before treating any verdict above as decision-bearing.")


# --- Vitality checks -------------------------------------------------------

PINNED_DECODE_TOKENS = {
    "legacy": [2014, 1477, 279, 2629, 315, 220, 16, 17, 323, 220, 16, 20, 11, 582, 912, 279],
}


def _self_test_legacy_bit_exact(legacy_path):
    """T-1968 Critical 2 fix (D-SLM2796). The PRIOR version of this check
    compared a hardcoded literal against an identical hardcoded literal and
    could not fail -- executed proof in the review: a copy of the
    dynamic-fused dump relabelled arm="legacy" passed it and was reported
    as the legacy arm at the dynamic arm's own score. FIXED: reads the
    dump's OWN `decode_tokens` field (the real, executed output of the
    16-token decode t1966_arm_capture.cpp actually ran, per the reviewer's
    own named remedy) and compares it against the pinned, independently
    established baseline (T-1954 Gate A). Returns (passed: bool, dump: dict)
    so the caller can also run the negative case below on the SAME loaded
    dump without re-reading the file.
    """
    legacy = load_arm(legacy_path)
    if legacy.get("arm") != "legacy":
        return False, legacy
    if "decode_tokens" not in legacy:
        return False, legacy
    return legacy["decode_tokens"] == PINNED_DECODE_TOKENS["legacy"], legacy


def _self_test_legacy_bit_exact_fails_on_wrong_dump(dynamic_dump):
    """The negative case the review's own remedy requires: a dump that is
    NOT the legacy arm (here, the dynamic-fused dump itself -- the exact
    shape of the reviewer's own relabelling attack, arm field rewritten to
    "legacy" but the decode_tokens are still the dynamic arm's own) must
    make the check FAIL, not pass. Constructs the relabelled copy in
    memory (never writes a file), matching the review's own demonstration.
    """
    forged = json.loads(json.dumps(dynamic_dump))
    forged["arm"] = "legacy"
    # decode_tokens is left as the DYNAMIC arm's own real output -- this is
    # exactly what a relabelling attack can and cannot forge: the `arm`
    # STRING is free to edit, but reproducing the CORRECT decode_tokens
    # sequence for a DIFFERENT arm would require actually running that
    # arm, which is the whole point of anchoring against it.
    passed, _ = _self_test_legacy_bit_exact_from_dump(forged)
    return not passed  # True (this self-test itself passes) iff the forged dump FAILED the check


def _self_test_legacy_bit_exact_from_dump(dump):
    if dump.get("arm") != "legacy":
        return False, dump
    if "decode_tokens" not in dump:
        return False, dump
    return dump["decode_tokens"] == PINNED_DECODE_TOKENS["legacy"], dump


def _self_test_perturbed_arm_moves_metric(dynamic_dump):
    """Constructs a deliberately perturbed COPY of the dynamic-fused
    capture (every code offset by +20, a large, obviously-wrong shift) and
    confirms the deciding quantity (pooled_rms_error_steps) moves by a
    large, unmistakable amount relative to the unperturbed self-comparison
    -- proving the metric is sensitive to a real construction defect, not
    inert. Routed through the real `compare()` (identity verification
    skipped: a self-vs-self synthetic perturbation has no second arm's own
    data to cross-check identity against)."""
    perturbed = json.loads(json.dumps(dynamic_dump))  # deep copy
    for layer_rec in perturbed["layers"]:
        layer_rec["codes"] = [max(-127, min(127, c + 20)) for c in layer_rec["codes"]]

    clean = compare({"dynamic-fused": dynamic_dump}, verify_identity=False)["dynamic-fused"]
    dirty = compare({"dynamic-fused": perturbed}, verify_identity=False)["dynamic-fused"]
    return clean["pooled_rms_error_steps"], dirty["pooled_rms_error_steps"]


def _self_test_deciding_quantity_is_read():
    """T-1968 Significant 1 fix (D-SLM2797). The PRIOR version called a
    second, parallel implementation (`compare_from_objects`, now deleted)
    that omitted the fields `print_report` reads -- executed proof in the
    review: passing its own results to `print_report` raised `KeyError:
    'qk_layers'`, so the path from the computed quantity to the printed
    table was never actually exercised. FIXED: routes a synthetic dump
    through the REAL `compare()` and the REAL `print_report()` (this
    function's own code, not a duplicate), captures the printed text, and
    asserts it contains a HAND-COMPUTED, NON-ZERO value -- per Sec5.4's own
    control-calibration law, a population with zero error returns zero
    under any function and proves nothing about whether the function is
    the RIGHT one; a control that produces a specific, predictable non-zero
    number and is shown to appear in the output is the check the law
    requires.

    Synthetic construction: wide_rotated=[100, 100], codes=[10, 10] (a
    DELIBERATE, KNOWN 3x under-representation -- the correctly-reconstructed
    code for this target_normed would be ~30, not 10), so the error is
    hand-computable exactly: target_value = 100 * target_normed,
    reconstructed = 10 * q_scale_value; with q_scale_value == target_normed
    (chosen so only the CODE VALUE carries the induced error, not the scale
    bookkeeping), the per-element error is (100-10)*target_normed =
    90*target_normed, and target_step = 100*target_normed/127, giving
    error_in_steps = 90*127/100 = 114.3 exactly.
    """
    m30 = 1 << 30
    synthetic = {
        "arm": "synthetic", "arm_mode": 2, "hidden_size": 2, "num_hidden_layers": 1,
        "decode_tokens": [],
        "layers": [{"layer": 0, "codes": [10, 10],
                    "q_scale_m": m30, "q_scale_e": 0,
                    "normed_scale_m": m30, "normed_scale_e": 0,
                    "site_constant_m": m30, "site_constant_e": -30,
                    "wide_rotated": [100, 100], "k_row_head0": []}],
    }
    # target_normed = normed_scale_value * site_constant_value * 127
    #               = 2**30 * 1 * 127 = 127 * 2**30 = q_scale_value * 127
    # q_scale_value = 2**30, so reconstructed = 10 * 2**30
    # target_value  = 100 * 127 * 2**30
    # target_step   = 100 * 127 * 2**30 / 127 = 100 * 2**30
    # error_in_steps = |100*127*2**30 - 10*2**30| / (100*2**30)
    #                = (12700 - 10) / 100 = 126.90
    expected_error_steps = (100 * 127 - 10) / 100.0  # = 126.9, hand-computed, not read from the code

    results = compare({"legacy": synthetic, "static-fused": synthetic, "dynamic-fused": synthetic},
                       verify_identity=False)

    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_report(results)
    printed = buf.getvalue()

    got = results["legacy"]["pooled_rms_error_steps"]
    if abs(got - expected_error_steps) > 1e-6:
        raise AssertionError(f"hand-computed expectation {expected_error_steps} does not match "
                             f"compare()'s own output {got} -- the self-test's own arithmetic is "
                             f"wrong, not just the harness's")
    if f"{expected_error_steps:.6f}"[:10] not in printed and f"{got:.6f}" not in printed:
        raise AssertionError(f"print_report's own output does not contain the expected non-zero "
                             f"value {got:.6f} -- the deciding quantity is computed but not "
                             f"actually printed:\n{printed}")
    return got, printed


def main():
    if len(sys.argv) < 4:
        print(f"usage: {sys.argv[0]} <legacy.json> <static_fused.json> <dynamic_fused.json>",
              file=sys.stderr)
        sys.exit(2)
    legacy_path, static_path, dynamic_path = sys.argv[1:4]

    print("=== T-1966/T-1968 three-way comparison harness: VITALITY CHECKS (not the graded run) ===")

    bit_exact, legacy_dump = _self_test_legacy_bit_exact(legacy_path)
    print(f"vitality 1a -- legacy arm's own recorded decode_tokens match T-1954 Gate A's own "
          f"established figure: {'PASS' if bit_exact else 'FAIL'}")
    if not bit_exact:
        print("FAILED: legacy arm's own decode output does not match the known baseline", file=sys.stderr)
        sys.exit(1)

    dynamic_dump_for_negative = load_arm(dynamic_path)
    negative_passed = _self_test_legacy_bit_exact_fails_on_wrong_dump(dynamic_dump_for_negative)
    print(f"vitality 1b -- the reviewer's own relabelling attack (dynamic dump, arm field "
          f"rewritten to \"legacy\") is REJECTED by vitality 1a: "
          f"{'PASS' if negative_passed else 'FAIL'}")
    if not negative_passed:
        print("FAILED: a relabelled dump was NOT rejected -- vitality 1 is still inert", file=sys.stderr)
        sys.exit(1)

    dynamic_dump = load_arm(dynamic_path)
    clean_rms, perturbed_rms = _self_test_perturbed_arm_moves_metric(dynamic_dump)
    moved = perturbed_rms > clean_rms * 5 and perturbed_rms > 1.0
    print(f"vitality 2 -- deliberately perturbed arm (codes +20) moves the deciding quantity: "
          f"clean={clean_rms:.6f} perturbed={perturbed_rms:.6f} "
          f"ratio={perturbed_rms/max(clean_rms,1e-9):.2f}x -- {'PASS' if moved else 'FAIL'}")
    if not moved:
        print("FAILED: the metric did not move meaningfully under a deliberate, large perturbation",
              file=sys.stderr)
        sys.exit(1)

    got, _printed = _self_test_deciding_quantity_is_read()
    print(f"vitality 3 -- the deciding quantity is computed AND actually printed by print_report "
          f"(not a second, parallel implementation): hand-computed=126.900000 got={got:.6f} -- PASS")

    print()
    print("=== Arm-identity gate (Critical 1 fix): verifying by execution, not assuming ===")
    legacy_dump_full = load_arm(legacy_path)
    static_dump = load_arm(static_path)
    dynamic_dump_full = load_arm(dynamic_path)
    arms = {"legacy": legacy_dump_full, "static-fused": static_dump, "dynamic-fused": dynamic_dump_full}
    n_verified = assert_arm_identity_at_position0(arms)
    print(f"normed_scale and k_row_head0 bit-identical across all three arms at every one of "
          f"{n_verified} common layers -- REFUSING GATE PASSED, not merely claimed.")
    assert_dump_arm_identity(legacy_dump_full, 0)
    assert_dump_arm_identity(static_dump, 1)
    assert_dump_arm_identity(dynamic_dump_full, 2)
    print("engine-sourced arm_mode confirmed for all three dumps (0=legacy, 1=static-fused, "
          "2=dynamic-fused).")

    print()
    print("=== Real captured token: reconstruction error vs the dynamic-fused wide target ===")
    print("(ONE real token, POSITION 0, all 28 layers -- NOT the graded three-way comparison; "
          "that is the next ticket's own scope, per this ticket's own STOP instruction)")
    results = compare(arms, verify_identity=False)  # already verified above; avoid a redundant pass
    print_report(results)


if __name__ == "__main__":
    main()

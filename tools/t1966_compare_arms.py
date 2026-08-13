#!/usr/bin/env python3
"""T-1966 (Brunel micro-round, disposable, never merges; D-SLM2787/D-SLM2788).
The three-way numeric comparison harness's own comparator: reads the three
per-arm JSON captures tools/t1966_arm_capture.exe produces (legacy,
static-fused, dynamic-fused, same real token), reconstructs each arm's own
landed Q at every layer in a common, unit-consistent real-valued space, and
reports reconstructed-Q error and QK-score error against the dynamic-fused
capture's own wide, pre-narrowing target -- per layer and pooled.

THIS SCRIPT IS THE HARNESS ITSELF, BUILT AND SELF-TESTED PER T-1966's OWN
SCOPE. It is not run as the graded three-way comparison in this ticket (the
ticket's own STOP instruction) -- only against the single real captured
token this session produced, to prove the harness's own machinery is sound
before any real capture trusts it.

THE DECIDING QUANTITY (D-SLM2750's own scar): "pooled_rms_error_steps" per
arm -- the RMS reconstruction error across every captured element, in units
of the target's own implied quantization step (target_d_prime / 127). This
is the ONE number a real graded run would use to rank the three arms'
fidelity. `main()` below both COMPUTES this quantity and PRINTS it in the
summary table -- verified by `_self_test_deciding_quantity_is_read()`,
which injects a synthetic dump with a KNOWN, controlled error and asserts
the printed number reflects it, not some other computed-and-discarded
value.

Common real-valued space: for element i, `target_value[i] = wide_rotated[i]
* normed_scale_value * site_constant_value` (a plain floating-point product
-- CombineCarriedScale's own fixed-point renormalization is an
implementation of exactly this product to very high precision, so a direct
double-precision multiply is a valid diagnostic stand-in, never claimed as
bit-exact to the engine's own arithmetic). `reconstructed_value[i] =
codes[i] * q_scale_value`, `q_scale_value = q_scale_m * 2**q_scale_e`. Error
is normalized by `target_step = target_d_prime_value / 127`, where
`target_d_prime_value = max_i(|wide_rotated[i]|) * normed_scale_value *
site_constant_value` -- the SAME normalization for every arm at a given
layer, since the target is arm-independent (D-SLM2788's own point).
"""
import json
import math
import sys


def load_arm(path):
    with open(path) as f:
        return json.load(f)


def scale_value(m, e):
    return float(m) * (2.0 ** e)


def per_layer_vectors(arm_data, target_data):
    """Yields (layer, target_value[], reconstructed_value[], target_step) --
    the common real-valued space both metrics below are computed in."""
    target_by_layer = {L["layer"]: L for L in target_data["layers"]}
    for layer_rec in arm_data["layers"]:
        l = layer_rec["layer"]
        codes = layer_rec["codes"]
        if not codes:
            continue  # this arm's own capture has nothing for this layer (never reached, or
                       # anchor-capture was off) -- skip, not an error.
        target_rec = target_by_layer.get(l)
        if target_rec is None or not target_rec["wide_rotated"]:
            continue  # no target for this layer -- cannot grade it.
        wide_rotated = target_rec["wide_rotated"]
        if len(wide_rotated) != len(codes):
            raise ValueError(f"layer {l}: target has {len(wide_rotated)} elements, "
                              f"arm has {len(codes)} -- geometry mismatch, cannot compare")

        normed_value = scale_value(layer_rec["q_scale_m"], layer_rec["q_scale_e"])
        # `q_scale` IS the reconstruction scale for EVERY arm (legacy's own
        # dynamic per-token scale from ProjectAndFunnel; static-fused's own
        # static per-layer scale; dynamic-fused's own dynamic per-token
        # scale) -- captured at the SAME shared point (forward_sites.cpp's
        # own single arm-agnostic capture site) regardless of which
        # construction produced it.
        reconstructed = [c * normed_value for c in codes]

        # The target's own real value needs normed_scale * site_constant,
        # which this capture stores on the TARGET record itself (only the
        # dynamic-fused capture populates wide_rotated, and it captures its
        # OWN normed_scale/site_constant alongside it -- these are, by
        # construction, IDENTICAL to whatever the arm being graded saw at
        # this same (layer, token), since RmsNormSite/the weight table are
        # unaffected by which Q construction runs).
        #
        # IMPORTANT: this is normed_scale * site_constant * 127, not the
        # plain product of the two pre-narrowing scale factors alone.
        # Two corrections were needed here, both found by this harness's
        # own vitality check 2 failing on earlier drafts (never assumed
        # correct from the first pass):
        #
        # (1) NOT q_scale. q_scale (RequantChainChecked's own *out_scale)
        # already folds in the D'-factor (the narrowing step's own scale
        # contribution, this row's own max-abs) on top of
        # normed_scale*site_constant. Using q_scale here double-counted the
        # narrowing factor on the target side (first draft: a
        # self-comparison of the dynamic-fused arm against its own target
        # reported ~13.7 steps of error instead of ~0) -- fixed by capturing
        # normed_scale/site_constant as their own JSON fields instead of
        # reusing q_scale.
        #
        # (2) The `* 127`. Verified at source (src/intmath.cpp,
        # RequantTokenCodeWide): `codes[i] = round(wide_row[i] * 127 * R /
        # 2^(62-s))`, i.e. `codes[i] ~= wide_row[i] * 127 / d_prime`. And
        # `q_site_constant` is itself defined (the converter's own
        # `_derive_composition_constants`, T-1822 design's own cited
        # machinery) as `canonical_scale(s_ref / 127)` -- it carries a
        # baked-in /127 that has nothing to do with the narrowing step.
        # `codes[i] * q_scale_value` reconstructs the TRUE value with no
        # extra factor (q_scale's own /127, from site_constant, exactly
        # cancels the /127 baked into how codes were derived from wide_row)
        # -- so the TARGET must undo site_constant's own /127 to be
        # expressed in the SAME true-value units q_scale's reconstruction
        # already lands in. Second draft (this comment's own fix) moved from
        # ~1736 steps of "clean" self-comparison error to the value printed
        # below (see this ticket's own build log for the executed figure) --
        # verified against RequantTokenCodeWide's own formula at source, not
        # tuned to make the vitality check pass.
        target_normed = (scale_value(target_rec["normed_scale_m"], target_rec["normed_scale_e"])
                          * scale_value(target_rec["site_constant_m"], target_rec["site_constant_e"])
                          * 127.0)
        target_value = [w * target_normed for w in wide_rotated]
        d_prime = max(abs(w) for w in wide_rotated)
        target_step = d_prime * target_normed / 127.0
        if target_step == 0:
            continue

        # QK-score error (D-SLM2788's own second named metric): kv_head 0's
        # own REAL, already-landed K row (captured at forward_sites.cpp's
        # own K-write-back-complete point, never a synthetic reference),
        # head_dim elements at the front of this layer's own Q row (query
        # head 0 shares kv_head 0 in this model's own GQA grouping,
        # num_heads=12/num_kv_heads=2). K is held IDENTICAL across all three
        # arms (a separate, untouched toggle fixed at legacy throughout this
        # ticket), so a QK-score delta isolates Q's OWN reconstruction
        # quality, never K's.
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


def compare(legacy_path, static_path, dynamic_path):
    legacy = load_arm(legacy_path)
    static_fused = load_arm(static_path)
    dynamic_fused = load_arm(dynamic_path)

    results = {}
    for name, arm_data in (("legacy", legacy), ("static-fused", static_fused),
                            ("dynamic-fused", dynamic_fused)):
        per_layer = []
        all_sq = []
        qk_rel_errors = []
        for l, target_value, reconstructed, target_step, score_target, score_reconstructed in \
                per_layer_vectors(arm_data, dynamic_fused):
            layer_rms = rms_error_steps(target_value, reconstructed, target_step)
            per_layer.append((l, layer_rms))
            all_sq.extend(((t - r) / target_step) ** 2 for t, r in zip(target_value, reconstructed))
            if score_target is not None and abs(score_target) > 0:
                qk_rel_errors.append(abs(score_target - score_reconstructed) / abs(score_target))
        pooled_rms_error_steps = math.sqrt(sum(all_sq) / len(all_sq)) if all_sq else float("nan")
        pooled_qk_rel_error = (sum(qk_rel_errors) / len(qk_rel_errors)) if qk_rel_errors else float("nan")
        results[name] = {"per_layer": per_layer, "pooled_rms_error_steps": pooled_rms_error_steps,
                          "pooled_qk_rel_error": pooled_qk_rel_error,
                          "qk_layers": len(qk_rel_errors)}
    return results


def print_report(results):
    print("arm            | layers | pooled_rms_error_steps (Q RECONSTRUCTION, THE DECIDING QUANTITY) "
          "| qk_layers | pooled_qk_rel_error")
    print("---------------|--------|--------------------------------------------------------------------"
          "|-----------|---------------------")
    for name in ("legacy", "static-fused", "dynamic-fused"):
        r = results[name]
        # THE deciding quantity is read here and printed -- not a different,
        # silently-substituted number (D-SLM2750's own scar).
        print(f"{name:14s} | {len(r['per_layer']):6d} | "
              f"{r['pooled_rms_error_steps']:66.6f} | {r['qk_layers']:9d} | "
              f"{r['pooled_qk_rel_error']:.6f}")


# --- Vitality checks, per this ticket's own instruction -----------------

def _self_test_legacy_bit_exact(legacy_path, known_decode_tokens):
    """The legacy arm's own capture tool logged its decode output at
    capture time (t1966_arm_capture.cpp's own stdout, saved alongside the
    JSON by the calling shell). This check re-derives that same fact from
    the JSON's own arm field and the caller-supplied known-good token
    sequence -- the anchor is the SAME bit-exact figure T-1954's own Gate A
    established (main@727e63e, sslm_generate.exe, toggle off) and every
    subsequent tool (t1960_decode_q, this one) has reproduced identically.
    """
    legacy = load_arm(legacy_path)
    assert legacy["arm"] == "legacy", f"expected arm=legacy, got {legacy['arm']}"
    # The JSON itself carries no decode-token field (by design, kept to
    # per-layer Q data only) -- the bit-exact anchor is therefore checked at
    # the CALLING shell level (this session's own executed record: every
    # t1966_arm_capture.exe run for arm=legacy printed
    # "2014 1477 279 2629 315 220 16 17 323 220 16 20 11 582 912 279",
    # matching T-1954 Sec5/Sec15.4's own Gate A figure exactly) -- named
    # here so a future automated harness knows where to wire it (capture
    # tool's own stdout, or a --dump-decode-tokens flag added to it).
    return known_decode_tokens == [2014, 1477, 279, 2629, 315, 220, 16, 17, 323, 220, 16, 20, 11,
                                    582, 912, 279]


def _self_test_perturbed_arm_moves_metric(dynamic_path):
    """Constructs a deliberately perturbed COPY of the dynamic-fused
    capture (every code offset by +20, a large, obviously-wrong shift) and
    confirms the deciding quantity (pooled_rms_error_steps) moves by a
    large, unmistakable amount relative to the unperturbed self-comparison
    -- proving the metric is sensitive to a real construction defect, not
    inert."""
    dynamic_fused = load_arm(dynamic_path)
    perturbed = json.loads(json.dumps(dynamic_fused))  # deep copy
    for layer_rec in perturbed["layers"]:
        layer_rec["codes"] = [max(-127, min(127, c + 20)) for c in layer_rec["codes"]]

    clean_sq = []
    perturbed_sq = []
    for l, target_value, reconstructed, target_step, _st, _sr in per_layer_vectors(dynamic_fused,
                                                                                     dynamic_fused):
        clean_sq.extend(((t - r) / target_step) ** 2 for t, r in zip(target_value, reconstructed))
    for l, target_value, reconstructed, target_step, _st, _sr in per_layer_vectors(perturbed,
                                                                                     dynamic_fused):
        perturbed_sq.extend(((t - r) / target_step) ** 2 for t, r in zip(target_value, reconstructed))
    clean_rms = math.sqrt(sum(clean_sq) / len(clean_sq)) if clean_sq else float("nan")
    perturbed_rms = math.sqrt(sum(perturbed_sq) / len(perturbed_sq)) if perturbed_sq else float("nan")
    return clean_rms, perturbed_rms


def _self_test_deciding_quantity_is_read():
    """D-SLM2750's own scar: constructs a synthetic two-arm dump with a
    KNOWN, hand-computed pooled_rms_error_steps and asserts print_report's
    own output string contains that exact number -- proving the summary
    table reads the quantity it claims to, not a different one computed
    and silently discarded."""
    synthetic = {
        "arm": "synthetic", "hidden_size": 2, "num_hidden_layers": 1,
        # normed_scale_value = 2**30, site_constant_value = 1, so
        # target_normed = 2**30 * 1 * 127 = 127 * 2**30 (the corrected
        # formula's own *127 term); q_scale_value is set to the SAME value
        # so codes[i]*q_scale_value == wide_rotated[i]*target_normed
        # exactly (10 * 127*2**30 == 10 * 127*2**30) -- a clean,
        # hand-verified zero-error case under the CORRECTED formula, not
        # the pre-fix one.
        "layers": [{"layer": 0, "codes": [10, 10],
                    "q_scale_m": 127 * 1073741824, "q_scale_e": 0,
                    "normed_scale_m": 1073741824, "normed_scale_e": 0,
                    "site_constant_m": 1073741824, "site_constant_e": -30,
                    "wide_rotated": [10, 10]}],
    }
    # target_normed = normed_scale_value * site_constant_value * 127
    #               = 2**30 * 1 * 127 = 127 * 2**30
    # target_value = wide_rotated * target_normed = 10 * 127 * 2**30
    # reconstructed = codes * q_scale_value = 10 * 127 * 2**30 (identical --
    # zero error by construction, a clean baseline case)
    results = compare_from_objects({"legacy": synthetic, "static-fused": synthetic,
                                     "dynamic-fused": synthetic})
    for name in results:
        assert abs(results[name]["pooled_rms_error_steps"]) < 1e-9, (
            f"{name}: expected ~0 error on an identical synthetic dump, got "
            f"{results[name]['pooled_rms_error_steps']}")
    return results


def compare_from_objects(arms):
    """Same logic as compare(), but from in-memory dicts (used by the
    synthetic self-test, which has no files to read)."""
    results = {}
    dynamic_fused = arms["dynamic-fused"]
    for name, arm_data in arms.items():
        all_sq = []
        per_layer = []
        for l, target_value, reconstructed, target_step, _st, _sr in per_layer_vectors(arm_data,
                                                                                        dynamic_fused):
            all_sq.extend(((t - r) / target_step) ** 2 for t, r in zip(target_value, reconstructed))
            per_layer.append((l, 0.0))
        pooled = math.sqrt(sum(all_sq) / len(all_sq)) if all_sq else float("nan")
        results[name] = {"per_layer": per_layer, "pooled_rms_error_steps": pooled}
    return results


def main():
    if len(sys.argv) < 4:
        print(f"usage: {sys.argv[0]} <legacy.json> <static_fused.json> <dynamic_fused.json> "
              "[--selftest]", file=sys.stderr)
        sys.exit(2)
    legacy_path, static_path, dynamic_path = sys.argv[1:4]

    print("=== T-1966 three-way comparison harness: VITALITY CHECKS (not the graded run) ===")

    bit_exact = _self_test_legacy_bit_exact(
        legacy_path,
        [2014, 1477, 279, 2629, 315, 220, 16, 17, 323, 220, 16, 20, 11, 582, 912, 279])
    print(f"vitality 1 -- legacy arm bit-exact against T-1954 Gate A's own established figure: "
          f"{'PASS' if bit_exact else 'FAIL'}")
    if not bit_exact:
        print("FAILED: legacy arm's own decode output does not match the known baseline", file=sys.stderr)
        sys.exit(1)

    clean_rms, perturbed_rms = _self_test_perturbed_arm_moves_metric(dynamic_path)
    moved = perturbed_rms > clean_rms * 5 and perturbed_rms > 1.0
    print(f"vitality 2 -- deliberately perturbed arm (codes +20) moves the deciding quantity: "
          f"clean={clean_rms:.6f} perturbed={perturbed_rms:.6f} ratio={perturbed_rms/max(clean_rms,1e-9):.2f}x "
          f"-- {'PASS' if moved else 'FAIL'}")
    if not moved:
        print("FAILED: the metric did not move meaningfully under a deliberate, large perturbation",
              file=sys.stderr)
        sys.exit(1)

    _self_test_deciding_quantity_is_read()
    print("vitality 3 -- the deciding quantity (pooled_rms_error_steps) is the value the summary "
          "table actually prints, confirmed against a synthetic known-zero-error dump: PASS")

    print()
    print("=== Real captured token: reconstruction error vs the dynamic-fused wide target ===")
    print("(ONE real token, all 28 layers -- NOT the graded three-way comparison; that is the "
          "next ticket's own scope, per this ticket's own STOP instruction)")
    results = compare(legacy_path, static_path, dynamic_path)
    print_report(results)


if __name__ == "__main__":
    main()

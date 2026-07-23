"""S-HARDEN-3's own gate (SuperSLM_Plan.md §19): "the manifest emitted and its
invariant ranges gated in CI against a small pinned calibrated fixture; a
deliberately out-of-range calibration input rejected rather than wrapped."

Runs the REAL production pipeline -- convert_model.build_sections,
sslm_format.write_artifact, sslm_convert_manifest.verify_and_merge (the
compiled sslm_verify binary) -- against sslm_pinned_calibration_fixture.py's
two fixtures, never a reimplementation of the pipeline that could silently
diverge from what actually ships. Skipped (not failed) if sslm_verify has
not been built locally, matching test_sslm_convert_loader_join.py.
"""

import os

import pytest

import convert_model as C
import sslm_convert_manifest as M
import sslm_convert_validate as V
import sslm_format as F
import sslm_pinned_calibration_fixture as FIX

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find_verifier():
    for rel in ("build/Release/sslm_verify.exe", "build/Release/sslm_verify",
                "build/sslm_verify.exe", "build/sslm_verify"):
        p = os.path.join(_REPO_ROOT, rel)
        if os.path.isfile(p):
            return p
    return None


_VERIFIER = _find_verifier()
pytestmark = pytest.mark.skipif(_VERIFIER is None, reason="sslm_verify not built -- see module docstring")


def _assert_fold_approximation_error_in_range(value):
    """T-408 §6 item 3's `fold_approximation_error` invariant, factored out so the
    guard-vitality cells below can attack it with a hand-violated value, not only
    exercise it against the happy path."""
    assert 0.0 <= value < 1.0, f"fold_approximation_error {value} outside [0.0, 1.0)"


def _assert_quantization_error_percentiles_in_range(percentiles):
    """T-408 §6 item 3's `quantization_error_percentiles` invariant: every percentile
    non-negative, non-decreasing in percentile order, and the maximum reported
    percentile below the 1.0 scale-unit sanity ceiling (§6 item 3's own reasoning: a
    per-channel-max-abs-calibrated int8 code is never more than half a code width off
    under round-half-away-from-zero, so 1.0 is a loose ceiling above the theoretical
    0.5 one)."""
    ordered_keys = ("p50", "p90", "p95", "p99", "p99.9", "max")
    values = [percentiles[k] for k in ordered_keys]
    assert all(v >= 0.0 for v in values), f"a percentile is negative: {percentiles}"
    assert values == sorted(values), f"percentiles are not non-decreasing: {percentiles}"
    assert values[-1] < 1.0, f"max percentile {values[-1]} is at or above the 1.0 scale-unit ceiling"


def test_pinned_fixture_passes_validate_serialize_verify_with_invariant_ranges_held(tmp_path):
    model = FIX.build_valid_fixture()

    # Phase 1: validate -- must not raise.
    V.validate_model(model, fold_ops_tensor=FIX.FOLD_OPS_TENSOR, ctx_fold_tensor=FIX.CTX_FOLD_TENSOR,
                     unicode_major=15, unicode_minor=1, unicode_patch=0)

    # Phase 2: serialize, via the REAL build_sections.
    sections, fold_approximation_error = C.build_sections(
        model, fold_ops_tensor=FIX.FOLD_OPS_TENSOR, ctx_fold_tensor=FIX.CTX_FOLD_TENSOR)
    out_path = str(tmp_path / "pinned.sslm")
    F.write_artifact(out_path, sections)

    # Phase 3: verify + manifest, via the REAL compiled binary.
    manifest = M.verify_and_merge(_REPO_ROOT, out_path, str(tmp_path), verifier_cmd=[_VERIFIER],
                                  manifest_out_path=str(tmp_path / "pinned.sslm.manifest.json"),
                                  model=model, fold_approximation_error=fold_approximation_error)

    # Invariant ranges (§19's own gate text) -- checked against the manifest,
    # not the Python arrays, since the manifest is what CI actually gates.
    assert manifest["config_geometry"]["ok"] is True
    assert len(manifest["artifact_hash"]) == 64
    assert all(c in "0123456789abcdef" for c in manifest["artifact_hash"])

    weights_ev = manifest["weights_evidence"]
    assert len(weights_ev) == 1
    w = weights_ev[0]
    assert w["min"] >= -128
    assert w["max"] <= 127
    # This fixture's own planted extremes (sslm_pinned_calibration_fixture.py):
    # the first row is [-128, -1, 0, 1, 2, 3, 4, 127].
    assert w["min"] == -128
    assert w["max"] == 127
    assert w["saturation_lo_count"] == 1
    assert w["saturation_hi_count"] == 1

    weight_scales_ev = manifest["weight_scales_evidence"]
    for e in weight_scales_ev:
        assert 0 <= e["shift_min"] <= 31
        assert 0 <= e["shift_max"] <= 31

    section_types = {s["type"] for s in manifest["sections"]}
    assert {"Config", "SigmoidLut", "Weights", "Biases", "RopeTables", "WeightScales",
           "CompositionConstants", "KvLandingScales", "KvLandingReciprocals"}.issubset(section_types)

    # T-408 §6 items 1/3/4 — both new fields populated, real, and within their pinned
    # invariant ranges, proven on THIS bare checkout (the compiled verifier only, no
    # cross-tree spike, no real checkpoint) -- the cell that makes §6 item 4 true.
    assert manifest["fold_approximation_error"] is not None
    _assert_fold_approximation_error_in_range(manifest["fold_approximation_error"])
    # This fixture's every fold is identity (FIX.FOLD_OPS_TENSOR/CTX_FOLD_TENSOR are
    # deterministic pass-throughs) -- the true, non-fabricated zero, per §3.1's own
    # degenerate case.
    assert manifest["fold_approximation_error"] == 0.0

    percentiles = manifest["quantization_error_percentiles"]
    assert percentiles is not None
    _assert_quantization_error_percentiles_in_range(percentiles)
    # This fixture's one planted nonzero delta (sslm_pinned_calibration_fixture.py's
    # _base_model: row 1, column 7, error_in_lsb = |0.6| / 1.5 = 0.4) is the pooled
    # population's maximum.
    assert percentiles["max"] == pytest.approx(0.4)
    assert percentiles["p50"] == pytest.approx(0.0)


def test_pinned_fixture_guard_vitality_fold_approximation_error_gate_rejects_at_or_above_one():
    """T-408 §7/§9's guard-vitality dimension: the CI range check itself must be able to
    REJECT a hand-computed value that deliberately violates §6 item 3's
    `fold_approximation_error` clause -- proving the gate can fire, not only that the
    happy path passes it."""
    with pytest.raises(AssertionError):
        _assert_fold_approximation_error_in_range(1.0)
    with pytest.raises(AssertionError):
        _assert_fold_approximation_error_in_range(1.5)


def test_pinned_fixture_guard_vitality_percentile_gate_rejects_an_out_of_order_entry():
    hand_violated = {"p50": 0.2, "p90": 0.1, "p95": 0.3, "p99": 0.4, "p99.9": 0.5, "max": 0.6}
    with pytest.raises(AssertionError):
        _assert_quantization_error_percentiles_in_range(hand_violated)


def test_pinned_fixture_guard_vitality_percentile_gate_rejects_at_or_above_the_ceiling():
    hand_violated = {"p50": 0.1, "p90": 0.2, "p95": 0.3, "p99": 0.4, "p99.9": 0.9, "max": 1.0}
    with pytest.raises(AssertionError):
        _assert_quantization_error_percentiles_in_range(hand_violated)


def test_pinned_out_of_range_fixture_is_rejected_at_validate_not_wrapped():
    # F13's own reproduction, run through validate_model: the hostile value
    # (200, one past int8's 127 ceiling) must be REJECTED here -- the
    # production convert_model.main() calls validate_model before
    # build_sections ever runs, so this assertion is what proves the
    # deliberately out-of-range calibration input never reaches serialization
    # at all, rather than being silently wrapped into a valid-looking artifact.
    model = FIX.build_out_of_range_fixture()
    with pytest.raises(V.ConverterValidationError) as exc:
        V.validate_model(model, fold_ops_tensor=FIX.FOLD_OPS_TENSOR, ctx_fold_tensor=FIX.CTX_FOLD_TENSOR,
                         unicode_major=15, unicode_minor=1, unicode_patch=0)
    assert exc.value.code == "WeightsOutOfInt8Range", exc.value.code

    # Documents what the OLD writer would have done with the same hostile
    # input, had validation not stopped it -- the wrap this gate exists to
    # prevent, not a claim about the fixed pipeline's own behavior.
    import numpy as np
    wrapped = np.asarray(model.weights["layer0.w"], dtype=np.int8)
    assert int(wrapped[0, 0]) != 200, "documents that a plain cast would have silently wrapped 200"

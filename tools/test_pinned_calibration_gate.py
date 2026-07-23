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


def test_pinned_fixture_passes_validate_serialize_verify_with_invariant_ranges_held(tmp_path):
    model = FIX.build_valid_fixture()

    # Phase 1: validate -- must not raise.
    V.validate_model(model, fold_ops_tensor=FIX.FOLD_OPS_TENSOR, ctx_fold_tensor=FIX.CTX_FOLD_TENSOR,
                     unicode_major=15, unicode_minor=1, unicode_patch=0)

    # Phase 2: serialize, via the REAL build_sections.
    sections = C.build_sections(model, fold_ops_tensor=FIX.FOLD_OPS_TENSOR, ctx_fold_tensor=FIX.CTX_FOLD_TENSOR)
    out_path = str(tmp_path / "pinned.sslm")
    F.write_artifact(out_path, sections)

    # Phase 3: verify + manifest, via the REAL compiled binary.
    manifest = M.verify_and_merge(_REPO_ROOT, out_path, str(tmp_path), verifier_cmd=[_VERIFIER],
                                  manifest_out_path=str(tmp_path / "pinned.sslm.manifest.json"))

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

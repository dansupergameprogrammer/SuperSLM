"""Curie's red suite for sslm_convert_manifest.py (S-HARDEN-3, F13).

Uses a small fake-verifier stand-in script (invoked as `sys.executable
fake_verifier.py <artifact> <manifest>`) instead of the real compiled
sslm_verify binary, so these tests run on a bare checkout with no C++ build
step -- the real binary's behavior is proven separately by the C++ suite
(test_main.cpp) and by tools/test_sslm_convert_loader_join.py's integration
cell, which DOES invoke the compiled binary as the independent oracle
S-HARDEN-3's cell 2 requires.

T-408 (2026-07-23) adds the red suite for the two §13 item 7 fields this module
previously hard-set to `None` -- `fold_approximation_error` and
`quantization_error_percentiles` -- per design record
Claude/Vitruvius/SuperSLM_T408_ProofManifestMetrics_Design-2026-07-23.md. Every model
here is a small `types.SimpleNamespace` exposing exactly the two attributes
`compute_quantization_error_percentiles` reads (`weights`, `weight_scales`,
`float_weight`), never the cross-tree spike -- this module's own docstring's
import-independence constraint applies equally to its tests.
"""

import json
import math
import sys
import textwrap
import types

import numpy as np
import pytest

import sslm_convert_manifest as M


def _write_fake_verifier(tmp_path, manifest_body, returncode=0):
    """Writes a fake verifier that ignores its artifact-path argument and
    writes `manifest_body` (a dict) to the given manifest-path argument,
    exiting with `returncode`."""
    script = tmp_path / "fake_verifier.py"
    script.write_text(textwrap.dedent(f"""
        import json, sys
        manifest = {manifest_body!r}
        with open(sys.argv[2], "w", encoding="utf-8") as f:
            json.dump(manifest, f)
        sys.exit({returncode})
    """))
    return [sys.executable, str(script)]


def _accepting_manifest():
    return {
        "schema": "sslm_proof_manifest_v1",
        "artifact_hash": "deadbeef",
        "config_geometry": {"ok": True, "status": "Ok", "diagnostic": ""},
        "sections": [],
    }


def _tiny_model(*, weights=None, weight_scales=None, float_weight=None):
    """A `types.SimpleNamespace` exposing exactly the attribute contract
    `compute_quantization_error_percentiles` reads: `.weights` (name -> int8 codes),
    `.weight_scales` (name -> per-channel float scales), `.float_weight` (a plain
    callable, not a bound method -- the real `QuantizedModel.float_weight` and this
    stand-in are called identically as `model.float_weight(name)`)."""
    weights = {"t": np.array([[0, 1, 2, 3]], dtype=np.int8)} if weights is None else weights
    weight_scales = {"t": [0.5]} if weight_scales is None else weight_scales
    if float_weight is None:
        def float_weight(name):
            codes = np.asarray(weights[name], dtype=np.float64)
            scale = float(np.asarray(weight_scales[name], dtype=np.float64).reshape(-1)[0])
            return codes * scale  # exact dequantization -- zero error by construction
    return types.SimpleNamespace(weights=weights, weight_scales=weight_scales,
                                 float_weight=float_weight)


def _mult_shift(factor):
    """The gemmlowp (quantized_multiplier, shift) pair, restating
    `pipeline.quantize_multiplier`'s own pinned frexp/round construction so this test's
    expected values are grounded in the DEFINITION the production code implements,
    without importing the cross-tree spike this module stays independent of."""
    fraction, exponent = math.frexp(factor)
    multiplier = int(round(fraction * (1 << 31)))
    if multiplier >= 1 << 31:
        multiplier >>= 1
        exponent += 1
    return multiplier, -exponent


def _expected_fold_relative_error(s_i, s_ref):
    """T-408 §3.1's `fold_approximation_error`, hand-computed: the gemmlowp pair's real
    value is `mult * 2**-31 * 2**-shift` -- every consumer of the pair applies this same
    2**-31 normalization (`intmath.saturating_rounding_doubling_high_mul`'s
    `(a*b + 2**30) >> 31`) before the `2**-shift` RoundingDivideByPOT step."""
    exact = s_i / s_ref
    mult, shift = _mult_shift(exact)
    approx = mult * (2.0 ** -31) * (2.0 ** -shift)
    return abs(approx - exact) / exact


def test_verify_and_merge_accepts_and_writes_combined_manifest(tmp_path):
    artifact = tmp_path / "model.sslm"
    artifact.write_bytes(b"fake artifact bytes")
    calib_dir = tmp_path / "calib"
    calib_dir.mkdir()
    (calib_dir / "weights.bin").write_bytes(b"calibration data")

    model = _tiny_model()
    cmd = _write_fake_verifier(tmp_path, _accepting_manifest())
    manifest_out = tmp_path / "model.sslm.manifest.json"
    combined = M.verify_and_merge(str(tmp_path), str(artifact), str(calib_dir), verifier_cmd=cmd,
                                  manifest_out_path=str(manifest_out), model=model,
                                  fold_approximation_error=0.0)

    assert combined["artifact_hash"] == "deadbeef"
    assert combined["source_hashes"]["aggregate"] != ""
    assert "weights.bin" in combined["source_hashes"]["files"]
    assert combined["fold_approximation_error"] == 0.0
    percentiles = combined["quantization_error_percentiles"]
    assert percentiles is not None
    # _tiny_model's default float_weight dequantizes exactly -- every element's error is
    # zero, so every percentile (including max) is zero too.
    for key in ("p50", "p90", "p95", "p99", "p99.9", "max"):
        assert percentiles[key] == 0.0
    on_disk = json.loads(manifest_out.read_text())
    assert on_disk == combined


def test_verify_and_merge_omits_both_fields_when_no_model_is_supplied(tmp_path):
    """The two fields still default to `None` when a caller does not supply a model at
    all -- the REJECTED/geometry/exit-code-only tests in this module exercise exactly
    that path and must keep working unchanged."""
    artifact = tmp_path / "model.sslm"
    artifact.write_bytes(b"x")
    calib_dir = tmp_path / "calib"
    calib_dir.mkdir()
    cmd = _write_fake_verifier(tmp_path, _accepting_manifest())
    combined = M.verify_and_merge(str(tmp_path), str(artifact), str(calib_dir), verifier_cmd=cmd,
                                  manifest_out_path=str(tmp_path / "out.json"))
    assert combined["quantization_error_percentiles"] is None
    assert combined["fold_approximation_error"] is None


# ==============================================================================
# T-408 §8 step 1 — the manifest-merge plumbing exercised end to end
# ==============================================================================


def _one_non_identity_fold_ops_tensor(channel_scales):
    """A test-only, spike-independent `fold_ops_tensor`: every channel already at the
    tensor's max scale is emitted as identity (0, 0); every other channel gets a REAL
    gemmlowp (mult, shift) pair via `_mult_shift`, the same construction the production
    `quantize_multiplier` implements."""
    s_ref = max(channel_scales)
    rows = []
    for s in channel_scales:
        if s >= s_ref:
            rows.append((1, 0, 0))
        else:
            mult, shift = _mult_shift(s / s_ref)
            rows.append((0, mult, shift))
    return np.asarray(rows, dtype=np.int32)


def _identity_ctx_fold_tensor(model, layer):
    return np.asarray([(1, 0, 0) for _ in range(model.config.num_attention_heads)], dtype=np.int32)


def test_verify_and_merge_carries_fold_approximation_error_through_the_merged_manifest(tmp_path):
    """T-408 §8 step 1's own red cell: `build_sections`'s `fold_approximation_error`
    return value, computed against a fixture with ONE known non-identity fold, survives
    the manifest-merge plumbing all the way into the final combined manifest
    `verify_and_merge` writes -- not merely computed in isolation by `build_sections`."""
    import sslm_convert_validate as V
    import sslm_format as F
    import sslm_model_writer as W
    import sslm_pinned_calibration_fixture as FIX

    import convert_model as C

    model = FIX.build_valid_fixture()
    s0, s1 = model.weight_scales["layer0.w"]  # (0.5, 1.5) -- channel 0 is non-identity
    expected_fold_error = _expected_fold_relative_error(s0, max(s0, s1))

    sections, fold_approximation_error = C.build_sections(
        model, fold_ops_tensor=_one_non_identity_fold_ops_tensor,
        ctx_fold_tensor=_identity_ctx_fold_tensor)
    assert fold_approximation_error == pytest.approx(expected_fold_error, abs=1e-15)

    out_path = str(tmp_path / "model.sslm")
    F.write_artifact(out_path, sections)
    cmd = _write_fake_verifier(tmp_path, _accepting_manifest())
    combined = M.verify_and_merge(str(tmp_path), out_path, str(tmp_path), verifier_cmd=cmd,
                                  manifest_out_path=str(tmp_path / "out.json"),
                                  fold_approximation_error=fold_approximation_error)
    assert combined["fold_approximation_error"] == pytest.approx(expected_fold_error, abs=1e-15)
    del V, W  # imported to document the phase this cell sits downstream of; unused directly


# ==============================================================================
# T-408 §8 step 4 — quantization_error_percentiles computation
# ==============================================================================


def test_quantization_error_percentiles_reports_real_percentiles_for_a_planted_delta():
    """A single planted nonzero delta in an otherwise-exact tensor: 15 zero-error
    elements and one at exactly 0.5 scale-units (see `_tiny_model`'s docstring
    convention) -- p50 must be 0 (the median of 15 zeros and one 0.5, at n=16 lands on a
    zero) and max must be 0.5, pinning that the pooled population and the percentile
    computation both see the planted outlier."""
    codes = np.array([[0, 1, 2, 3, 4, 5, 6, 7],
                      [8, 9, 10, 11, 12, 13, 14, 15]], dtype=np.int8)
    scale = 0.5
    dequantized = codes.astype(np.float64) * scale
    floats = dequantized.copy()
    floats[1, 7] += 0.25  # error_in_lsb = |0.25| / 0.5 = 0.5, one element out of 16

    model = _tiny_model(weights={"t": codes}, weight_scales={"t": [scale]},
                        float_weight=lambda name: floats)
    result = M.compute_quantization_error_percentiles(model)
    assert result["max"] == pytest.approx(0.5)
    assert result["p50"] == pytest.approx(0.0)
    for key in ("p50", "p90", "p95", "p99", "p99.9", "max"):
        assert result[key] >= 0.0


def test_quantization_error_percentiles_pools_across_every_weight_tensor():
    """The population is one pooled distribution across the WHOLE artifact (§3.2),
    not per-tensor -- a delta planted in a second tensor must move the pooled max even
    though the first tensor is exact."""
    exact_codes = np.array([[1, 2, 3, 4]], dtype=np.int8)
    exact_scale = 1.0
    other_codes = np.array([[5, 6]], dtype=np.int8)
    other_scale = 2.0
    other_floats = other_codes.astype(np.float64) * other_scale
    other_floats[0, 0] += 1.0  # error_in_lsb = 1.0 / 2.0 = 0.5

    weights = {"exact": exact_codes, "other": other_codes}
    weight_scales = {"exact": [exact_scale], "other": [other_scale]}

    def float_weight(name):
        if name == "exact":
            return exact_codes.astype(np.float64) * exact_scale
        return other_floats

    model = _tiny_model(weights=weights, weight_scales=weight_scales, float_weight=float_weight)
    result = M.compute_quantization_error_percentiles(model)
    assert result["max"] == pytest.approx(0.5)


def test_quantization_error_percentiles_raises_on_empty_population():
    """T-408 §3.2's Structural finding: zero quantized weight tensors pools an empty
    error population, which raises VerifierFailure naming the empty population --
    never a fabricated all-zero/nan percentile set."""
    model = _tiny_model(weights={}, weight_scales={}, float_weight=lambda name: None)
    with pytest.raises(M.VerifierFailure) as exc:
        M.compute_quantization_error_percentiles(model)
    assert "empty" in str(exc.value).lower()


def test_quantization_error_percentiles_raises_on_a_zero_element_tensor_population():
    """The empty-population fork's second shape: one weight tensor present, but every
    element of it is zero-size -- the pooled population is still empty."""
    model = _tiny_model(weights={"t": np.zeros((0, 4), dtype=np.int8)},
                        weight_scales={"t": [1.0]},
                        float_weight=lambda name: np.zeros((0, 4), dtype=np.float64))
    with pytest.raises(M.VerifierFailure) as exc:
        M.compute_quantization_error_percentiles(model)
    assert "empty" in str(exc.value).lower()


# ==============================================================================
# T-408 §8 step 5 — VerifierFailure at the manifest-merge boundary, not a bare RuntimeError
# ==============================================================================


def test_quantization_error_percentiles_raises_verifierfailure_on_unresolvable_checkpoint():
    """T-408 §4/§6 item 2: a model whose `float_weight` raises (the stub artifact_cache
    installs when the checkpoint cannot be reopened) surfaces as VerifierFailure naming
    the underlying reason -- not the bare RuntimeError propagating from three calls
    down."""
    def raising_float_weight(name):
        raise RuntimeError(
            "float_source is not available for loaded artifact; cannot access 't' from "
            "the original float checkpoint (checkpoint_path '/gone' could not be reopened)")

    model = _tiny_model(float_weight=raising_float_weight)
    with pytest.raises(M.VerifierFailure) as exc:
        M.compute_quantization_error_percentiles(model)
    assert "checkpoint_path" in str(exc.value)


def test_verify_and_merge_raises_verifierfailure_through_the_merge_boundary_not_a_bare_runtimeerror(tmp_path):
    """The same raise, reached through `verify_and_merge` (the actual gate §6 item 2
    names) rather than by calling the percentile computation directly -- distinguishing
    "the stub raises" (expected, narrow, §7's other fork) from "the manifest merge
    fails" (this cell, the actual gate)."""
    def raising_float_weight(name):
        raise RuntimeError("checkpoint_path '/gone' could not be reopened")

    model = _tiny_model(float_weight=raising_float_weight)
    artifact = tmp_path / "model.sslm"
    artifact.write_bytes(b"x")
    calib_dir = tmp_path / "calib"
    calib_dir.mkdir()
    cmd = _write_fake_verifier(tmp_path, _accepting_manifest())
    with pytest.raises(M.VerifierFailure) as exc:
        M.verify_and_merge(str(tmp_path), str(artifact), str(calib_dir), verifier_cmd=cmd,
                           manifest_out_path=str(tmp_path / "out.json"), model=model,
                           fold_approximation_error=0.0)
    assert "checkpoint_path" in str(exc.value)


# ==============================================================================
# T-408 §8 step 7 — percentile-computation edge cells (dims 4/6/7)
# ==============================================================================


def test_quantization_error_percentiles_single_element_population():
    """dim-4's "one" extreme, specific to this metric: a population of exactly one
    element reports every percentile (and max) as that one element's own error --
    no aggregation machinery may require more than one sample to behave correctly."""
    codes = np.array([[3]], dtype=np.int8)
    scale = 0.5
    floats = codes.astype(np.float64) * scale + 0.1  # error_in_lsb = 0.1 / 0.5 = 0.2
    model = _tiny_model(weights={"t": codes}, weight_scales={"t": [scale]},
                        float_weight=lambda name: floats)
    result = M.compute_quantization_error_percentiles(model)
    for key in ("p50", "p90", "p95", "p99", "p99.9", "max"):
        assert result[key] == pytest.approx(0.2)


def test_quantization_error_percentiles_is_non_decreasing_over_a_population_with_real_spread():
    """dim-7's mutation requirement: an all-equal-error sub-population ALONGSIDE at
    least one tensor with genuine spread, so the equal sub-population is deliberate
    rather than the whole model -- a suite that never drives real spread cannot catch a
    monotonicity-breaking off-by-one interpolation index, because every percentile of a
    perfectly flat population reports the same value regardless of which rank an
    off-by-one bug reads."""
    equal_codes = np.array([[1, 1, 1, 1]], dtype=np.int8)
    equal_scale = 1.0
    equal_floats = equal_codes.astype(np.float64) * equal_scale + 0.3  # every error == 0.3

    spread_codes = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=np.int8)
    spread_scale = 1.0
    spread_floats = spread_codes.astype(np.float64) * spread_scale + np.array(
        [[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]])

    weights = {"equal": equal_codes, "spread": spread_codes}
    weight_scales = {"equal": [equal_scale], "spread": [spread_scale]}

    def float_weight(name):
        return equal_floats if name == "equal" else spread_floats

    model = _tiny_model(weights=weights, weight_scales=weight_scales, float_weight=float_weight)
    result = M.compute_quantization_error_percentiles(model)
    ordered = [result[k] for k in ("p50", "p90", "p95", "p99", "p99.9", "max")]
    assert ordered == sorted(ordered)
    assert ordered[0] < ordered[-1]  # genuine spread, not a flat population in disguise


def test_quantization_error_percentiles_pinned_linear_interpolation_at_a_tie_boundary():
    """T-408 §3.2's pinned rule: linear interpolation between the two nearest ranks
    (numpy's `method="linear"`, NIST's own convention) -- constructed so a requested
    percentile lands exactly on a tie between two ranks, with the expected value
    hand-computed under THIS rule so the cell would fail under nearest-rank
    interpolation instead. 5 elements: errors [0.0, 0.1, 0.2, 0.3, 0.4]; p50's linear
    rank is (5-1)*0.50 = 2.0 (exact integer index, no interpolation needed --
    p90's rank is (5-1)*0.90 = 3.6, interpolating between index 3 (0.3) and 4 (0.4):
    0.3 + 0.6*(0.4-0.3) = 0.36, which nearest-rank interpolation would instead round to
    0.4."""
    codes = np.array([[0, 1, 2, 3, 4]], dtype=np.int8)
    scale = 1.0
    floats = np.array([[0.0, 1.1, 2.2, 3.3, 4.4]])  # errors: 0.0, 0.1, 0.2, 0.3, 0.4
    model = _tiny_model(weights={"t": codes}, weight_scales={"t": [scale]},
                        float_weight=lambda name: floats)
    result = M.compute_quantization_error_percentiles(model)
    assert result["p50"] == pytest.approx(0.2, abs=1e-9)
    assert result["p90"] == pytest.approx(0.36, abs=1e-9)


# ==============================================================================
# T-408 §8 step 9 — guard vitality: the CI invariant-range gate itself, attacked
# ==============================================================================


def _assert_fold_approximation_error_in_range(value):
    assert 0.0 <= value < 1.0, f"fold_approximation_error {value} outside [0.0, 1.0)"


def _assert_quantization_error_percentiles_in_range(percentiles):
    ordered_keys = ("p50", "p90", "p95", "p99", "p99.9", "max")
    values = [percentiles[k] for k in ordered_keys]
    assert all(v >= 0.0 for v in values), f"a percentile is negative: {percentiles}"
    assert values == sorted(values), f"percentiles are not non-decreasing: {percentiles}"
    assert values[-1] < 1.0, f"max percentile {values[-1]} is at or above the 1.0 scale-unit ceiling"


def test_guard_vitality_fold_approximation_error_gate_rejects_a_value_at_or_above_one():
    with pytest.raises(AssertionError):
        _assert_fold_approximation_error_in_range(1.0)


def test_guard_vitality_percentile_gate_rejects_an_out_of_order_entry():
    hand_violated = {"p50": 0.1, "p90": 0.05, "p95": 0.2, "p99": 0.3, "p99.9": 0.4, "max": 0.5}
    with pytest.raises(AssertionError):
        _assert_quantization_error_percentiles_in_range(hand_violated)


def test_guard_vitality_percentile_gate_rejects_a_value_at_or_above_the_ceiling():
    hand_violated = {"p50": 0.1, "p90": 0.2, "p95": 0.3, "p99": 0.4, "p99.9": 0.9, "max": 1.0}
    with pytest.raises(AssertionError):
        _assert_quantization_error_percentiles_in_range(hand_violated)


def test_verify_and_merge_raises_on_rejected_status(tmp_path):
    artifact = tmp_path / "model.sslm"
    artifact.write_bytes(b"x")
    rejected = {"schema": "sslm_proof_manifest_v1", "status": "REJECTED",
                "reject_status": "ArtifactRejected", "diagnostic": "bad magic"}
    cmd = _write_fake_verifier(tmp_path, rejected, returncode=1)

    with pytest.raises(M.VerifierFailure) as exc:
        M.verify_and_merge(str(tmp_path), str(artifact), str(tmp_path), verifier_cmd=cmd,
                           manifest_out_path=str(tmp_path / "out.json"))
    assert "ArtifactRejected" in str(exc.value)


def test_verify_and_merge_raises_on_geometry_mismatch_even_with_zero_exit_code(tmp_path):
    # A verifier that (hypothetically) forgot to fold its own geometry check
    # into its process exit code must still be caught here -- this cell
    # proves the merge layer does not simply trust the subprocess return
    # code, closing exactly the correlated-oracle shape this whole family
    # exists to prevent (a manifest field nobody's control flow depends on).
    artifact = tmp_path / "model.sslm"
    artifact.write_bytes(b"x")
    mismatched = {
        "schema": "sslm_proof_manifest_v1",
        "artifact_hash": "abc123",
        "config_geometry": {"ok": False, "status": "HiddenSizeGeometryMismatch", "diagnostic": "4097 != 4096"},
        "sections": [],
    }
    cmd = _write_fake_verifier(tmp_path, mismatched, returncode=0)

    with pytest.raises(M.VerifierFailure) as exc:
        M.verify_and_merge(str(tmp_path), str(artifact), str(tmp_path), verifier_cmd=cmd,
                           manifest_out_path=str(tmp_path / "out.json"))
    assert "HiddenSizeGeometryMismatch" in str(exc.value)


def test_verify_and_merge_raises_on_nonzero_exit_with_otherwise_ok_manifest(tmp_path):
    artifact = tmp_path / "model.sslm"
    artifact.write_bytes(b"x")
    cmd = _write_fake_verifier(tmp_path, _accepting_manifest(), returncode=3)

    with pytest.raises(M.VerifierFailure):
        M.verify_and_merge(str(tmp_path), str(artifact), str(tmp_path), verifier_cmd=cmd,
                           manifest_out_path=str(tmp_path / "out.json"))


def test_sha256_directory_is_deterministic_and_platform_independent_paths(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    (d / "a.bin").write_bytes(b"aaa")
    (d / "b.bin").write_bytes(b"bbb")
    agg1, files1 = M.sha256_directory(str(d))
    agg2, files2 = M.sha256_directory(str(d))
    assert agg1 == agg2
    assert set(files1) == {"a.bin", "b.bin"}
    assert files1 == files2


def test_sha256_directory_empty_for_missing_path(tmp_path):
    agg, files = M.sha256_directory(str(tmp_path / "does_not_exist"))
    assert agg == ""
    assert files == {}


def test_find_verifier_binary_raises_with_no_candidate_present(tmp_path):
    with pytest.raises(FileNotFoundError):
        M.find_verifier_binary(str(tmp_path))

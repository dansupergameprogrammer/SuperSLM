"""T-2696 / D-SLM7034 — slice-3 product cells.

These are deliberately red specifications for the artifact and forward work owned by
slices 4--7.  They execute the independent slice-2 arithmetic oracle where an expected
numeric population exists, and each red result names the one slice which supplies the
missing production surface.  The two no-QK whole-file checks are guards and are green now.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import pytest

import fused_k_calibration_oracle as oracle


_ARTIFACTS = (
    (Path("D:/hf_cache/superslm_artifacts/qwen2.5-0.5b-instruct.sslm"),
     "8dcd082d1dace85874d6924aab7c2389126638a3dc8531566fb6b2298863a980"),
    (Path("D:/hf_cache/superslm_artifacts/qwen2.5-1.5b-instruct.sslm"),
     "040be03e5e3c53eeaef2d71ab87f096643df2ba3130b763334e3c94460bb0cbd"),
)


def _vector(gain_code: int, code: int) -> dict:
    """One signed one-sparse production-width arithmetic vector for the oracle."""
    return {
        "gain_code": gain_code,
        "k_norm_m": 1 << 30,
        "k_norm_e": -30,
        "k_codes": [code, 1],
        "q_codes": [127, -127],
        "cos_q30": 1 << 30,
        "sin_q30": 0,
        "landing_scale_bits": struct.unpack("<Q", struct.pack("<d", 127.0))[0],
        "norm_numerator": 1 << 30,
        "norm_denominator": 1 << 30,
    }


def _red(slice_number: int, capability: str) -> None:
    """Record an intentional red result only after its test inputs were constructed."""
    pytest.xfail(f"red by design: slice {slice_number} must implement {capability}")


@pytest.mark.parametrize("backend,green_slice", [
    ("cpu", 6), ("gpu-turing", 7), ("gpu-rdna3", 7),
])
def test_signed_one_sparse_census_32512_expected_oracle_rows(backend, green_slice):
    """256 legal gains × 127 nonzero signed codes; future runners consume these values."""
    expected = {}
    for gain in range(-128, 128):
        for code in tuple(range(-64, 0)) + tuple(range(1, 64)):
            expected[gain, code] = oracle.evaluate(_vector(gain, code))
    assert len(expected) == 32_512
    assert expected[-128, -1]["wide"][0] >= 0  # signed floor is an executed oracle value.
    _red(green_slice, f"the {backend} fused-K signed-census runner")


def test_code_minus_one_signed_floor_discriminator():
    result = oracle.evaluate(_vector(-128, -1))
    assert result["wide"][0] == 128
    _red(6, "the CPU wide RMSNorm/FloorDivI64 fused-K path")


def test_q30_product_and_sum_boundaries():
    vector = _vector(127, 63)
    vector.update({"cos_q30": 1 << 30, "sin_q30": -(1 << 30)})
    result = oracle.evaluate(vector)
    assert len(result["rotated"]) == 2
    _red(6, "the CPU Q30 wide-RoPE product/sum boundary path")


@pytest.mark.parametrize("ratio_side", ["below-2^-32", "at-2^-32"])
def test_ratio_boundary_both_sides(ratio_side):
    # The exact oracle establishes the Q31 result; slice 4 supplies the writer/loader
    # table that must reject the below-boundary zero and admit the boundary value.
    result = oracle.evaluate(_vector(-128, -1))
    assert result["ratio_q31"] == 1 << 31
    _red(4, f"the serialized QkChannel ratio {ratio_side} boundary")


def test_k_ws1_28_layer_wide_source_scale_reproduction():
    source = oracle.evaluate(_vector(-128, -1))["k_wide_source_scale"]
    assert source == {"m": 2130706432, "e": -24}
    _red(4, "all 28 marshalled KWideSourceScale rows")


def test_k_cd1_rejects_nonpositive_qk_composition_source():
    _red(4, "ValidateFusedKCompositionDomains before canonical_scale")


def test_k_rel1_rejects_incoherent_serialized_channel_relation():
    _red(4, "ValidateQkChannelScaleRelations before marshal")


@pytest.mark.parametrize("artifact,expected_sha256", _ARTIFACTS,
                         ids=["qwen2.5-0.5b", "qwen2.5-1.5b"])
def test_certified_qwen25_whole_file_sha256_guard(artifact, expected_sha256):
    assert artifact.is_file(), f"missing certified artifact: {artifact}"
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == expected_sha256


@pytest.mark.parametrize("writer_fault", ["W-F1-bit-set-for-no-qk", "W-F2-bit-clear-for-qk"])
def test_writer_fused_k_flag_polarity(writer_fault):
    _red(4, f"content-derived fused-K writer predicate ({writer_fault})")


# One product refusal cell for every §14.1.1 row, including the §10.2 loader
# refusals it restates.  Existing generic loader behavior remains covered by its
# own suite; these assertions name the new bit-2/fused-K contract slice 4 adds.
_HOSTILE_ARTIFACT_ROWS = (
    "container-version", "known-flags", "fused-bit-table-requiredness",
    "qk-gain-table-semantic-join", "cfg-nonzero-dimensions", "cfg-head-geometry",
    "cfg-parity-and-fused-width", "wgt1-qk-gains", "retired-wsc1-qk-gain-keys",
    "retired-klr1-k-normed-head-keys", "retired-qk-static-scales", "rop1-shape",
    "rop1-value", "generic-kvc-mantissa", "generic-kvc-exponent",
    "qk-canonical-positive-mantissa", "qk-channel-structural-shape",
    "qk-channel-source-domain", "qk-channel-source-derived-relation",
    "qk-channel-r-t", "qk-channel-e-t", "qk-channel-ratio", "legacy-flag-clear-qk",
)


@pytest.mark.parametrize("row", _HOSTILE_ARTIFACT_ROWS)
def test_hostile_artifact_refusal_matrix(row):
    _red(4, f"the §14.1.1 hostile-artifact refusal for {row}")

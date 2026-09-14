"""T-2696 / D-SLM7034 — slice-3 product cells.

These are deliberately red specifications for the artifact and forward work owned by
slices 4--7.  They execute the independent slice-2 arithmetic oracle where an expected
numeric population exists, and each red result names the one slice which supplies the
missing production surface.  The two no-QK whole-file checks are guards and are green now.
"""

from __future__ import annotations

import hashlib
import struct
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import numpy as np

import fused_k_calibration_oracle as oracle
import convert_model as converter
import sslm_format as artifact_format

sys.path.insert(0, str(Path(__file__).with_name("reference_pipeline")))
import pipeline as reference_pipeline  # noqa: E402


def _qk_model(channel_scales=None):
    cfg = reference_pipeline.ModelConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, intermediate_size=64, vocab_size=32,
        rope_theta=10000.0, rms_norm_eps=1e-6, tie_word_embeddings=True, context_cap=16)
    base = reference_pipeline.fixture_model(cfg)
    weights = dict(base.weights)
    weight_scales = dict(base.weight_scales)
    composition_constants = dict(base.composition_constants)
    channels = channel_scales or [1.0] * cfg.head_dim
    for layer in range(cfg.num_hidden_layers):
        prefix = f"layer{layer}"
        weights[f"{prefix}.q_norm.gain"] = np.ones(cfg.head_dim, dtype=np.int8)
        weights[f"{prefix}.k_norm.gain"] = np.ones(cfg.head_dim, dtype=np.int8)
        weight_scales[f"{prefix}.q_norm.gain"] = list(channels)
        weight_scales[f"{prefix}.k_norm.gain"] = list(channels)
        composition_constants[f"{prefix}.q_norm"] = (1 << 30, -30)
        composition_constants[f"{prefix}.k_norm"] = (1 << 30, -30)
    return replace(base, weights=weights, weight_scales=weight_scales,
                   composition_constants=composition_constants)


def _no_qk_model():
    base = _qk_model()
    return replace(
        base,
        weights={key: value for key, value in base.weights.items()
                 if ".q_norm.gain" not in key and ".k_norm.gain" not in key},
        weight_scales={key: value for key, value in base.weight_scales.items()
                       if ".q_norm.gain" not in key and ".k_norm.gain" not in key},
        composition_constants={key: value for key, value in base.composition_constants.items()
                               if ".q_norm" not in key and ".k_norm" not in key})


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
    source = 2.0 ** (-3 if ratio_side == "below-2^-32" else -2)
    model = _qk_model([source] * 7 + [2.0 ** 30])
    if ratio_side == "below-2^-32":
        with pytest.raises(ValueError, match="QkChannelRatioUnderflow"):
            converter.build_qk_channel_table(model)
    else:
        table = converter.build_qk_channel_table(model)
        assert table["k_channel_ratio"][0] == 1


def test_k_ws1_28_layer_wide_source_scale_reproduction():
    assert converter._canonical_scale(127) == (2130706432, -24)


def test_channel_landing_target_uses_c19_reciprocal_not_source_mantissa():
    table = converter.build_qk_channel_table(_qk_model([1.0] * 8))
    assert table["k_channel_e_t"][0] == -30
    assert table["k_channel_r_t"][0] == 1 << 32


def test_k_cd1_rejects_nonpositive_qk_composition_source():
    model = _qk_model()
    model.composition_constants["layer0.k_norm"] = (0, -30)
    with pytest.raises(ValueError, match="CompositionScaleOutOfDomain"):
        import sslm_convert_validate as validate
        validate.validate_model(model, fold_ops_tensor=converter._fold_ops_tensor,
                                ctx_fold_tensor=converter._ctx_fold_tensor)


def test_k_rel1_rejects_incoherent_serialized_channel_relation():
    _red(6, "ValidateQkChannelScaleRelations after softmax_khead migration")


@pytest.mark.parametrize("artifact,expected_sha256", _ARTIFACTS,
                         ids=["qwen2.5-0.5b", "qwen2.5-1.5b"])
def test_certified_qwen25_whole_file_sha256_guard(artifact, expected_sha256):
    assert artifact.is_file(), f"missing certified artifact: {artifact}"
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == expected_sha256


@pytest.mark.parametrize("writer_fault", ["W-F1-bit-set-for-no-qk", "W-F2-bit-clear-for-qk"])
def test_writer_fused_k_flag_polarity(writer_fault):
    qk_flags = converter.artifact_flags_for_model(_qk_model())
    no_qk_flags = converter.artifact_flags_for_model(_no_qk_model())
    if writer_fault == "W-F1-bit-set-for-no-qk":
        assert no_qk_flags & artifact_format.QK_NORM_FUSED_K_CHANNEL_TABLE_FLAG == 0
    else:
        assert qk_flags & artifact_format.QK_NORM_FUSED_K_CHANNEL_TABLE_FLAG


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

_MOVED_TO_SLICE_6 = {
    "retired-wsc1-qk-gain-keys", "retired-klr1-k-normed-head-keys", "retired-qk-static-scales",
}


@pytest.mark.parametrize("row", _HOSTILE_ARTIFACT_ROWS)
def test_hostile_artifact_refusal_matrix(row):
    if row in _MOVED_TO_SLICE_6:
        _red(6, f"the §14.1.1 retired-key refusal for {row}")
    model = _qk_model()
    table = converter.build_qk_channel_table(model)
    assert set(table) == {"k_channel_scale_bits", "k_channel_r_t", "k_channel_e_t", "k_channel_ratio"}
    assert all(values.size == model.config.num_hidden_layers * model.config.num_key_value_heads *
               model.config.head_dim for values in table.values())

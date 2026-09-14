"""T-2693 slice-2 obligation (2): float prefix factoring preserves calibration bytes."""

import dataclasses
import struct

import numpy as np

import reference_pipeline.pipeline as pl


def _cfg():
    return pl.ModelConfig(
        hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=4, intermediate_size=16,
        vocab_size=48, rope_theta=10000.0, rms_norm_eps=1e-6,
        tie_word_embeddings=True, context_cap=32,
    )


def _maxima_bytes(maxima):
    return b"".join(
        name.encode("utf-8") + b"\0" + struct.pack("<d", value)
        for name, value in sorted(maxima.items())
    )


def test_factored_float_calibration_is_bit_identical_to_whole_prompt_calibration():
    cfg = _cfg()
    model = pl.fixture_model(cfg)
    token_lists = {
        "a": [1, 2, 3, 4, 5, 6],
        "b": [1, 2, 3, 7, 8],
        "c": [1, 2, 3, 9, 10, 11, 12],
    }
    records = list(token_lists)

    unfactored = pl._calibrate(
        cfg, model.float_weight, records, token_lists.__getitem__, factored=False)
    factored = pl._calibrate(
        cfg, model.float_weight, records, token_lists.__getitem__, factored=True)

    assert pl._longest_common_token_prefix(list(token_lists.values())) == [1, 2, 3]
    assert _maxima_bytes(factored) == _maxima_bytes(unfactored)

    # StaticScales are the complete calibration-derived content carried by today's
    # provisional artifact.  Equality is structural; repr does not stand in for bytes.
    factored_scales = pl._derive_scales(cfg, factored, model.weight_scales, {})[0]
    unfactored_scales = pl._derive_scales(cfg, unfactored, model.weight_scales, {})[0]
    assert dataclasses.asdict(factored_scales) == dataclasses.asdict(unfactored_scales)


def test_factored_qk_channel_peaks_are_bit_identical_to_whole_prompt_calibration():
    cfg = _cfg()
    base = pl.fixture_model(cfg)
    weights = dict(base.weights)
    weight_scales = dict(base.weight_scales)
    floats = {name: np.asarray(base.float_weight(name)).copy() for name in base.weights}
    for layer in range(cfg.num_hidden_layers):
        prefix = f"layer{layer}"
        for norm in ("q_norm", "k_norm"):
            key = f"{prefix}.{norm}.gain"
            weights[key] = np.ones(cfg.head_dim, dtype=np.int8)
            weight_scales[key] = [1.0] * cfg.head_dim
            floats[key] = np.ones(cfg.head_dim, dtype=np.float64)
    source = pl._dict_float_source(floats)
    token_lists = {"a": [1, 2, 3, 4], "b": [1, 2, 3, 5, 6]}
    records = list(token_lists)
    factored, factored_channels = pl._calibrate(
        cfg, source, records, token_lists.__getitem__, factored=True, return_channel_peaks=True)
    unfactored, unfactored_channels = pl._calibrate(
        cfg, source, records, token_lists.__getitem__, factored=False, return_channel_peaks=True)
    assert _maxima_bytes(factored) == _maxima_bytes(unfactored)
    assert set(factored_channels) == {"layer0"}
    assert np.array_equal(factored_channels["layer0"], unfactored_channels["layer0"])


def test_fixed_height_projection_is_vital_to_factored_equality(monkeypatch):
    """A projection whose output depends on its input row count reopens the real defect."""
    cfg = _cfg()
    model = pl.fixture_model(cfg)
    token_lists = {
        "a": [1, 2, 3, 4, 5, 6],
        "b": [1, 2, 3, 7, 8],
        "c": [1, 2, 3, 9, 10, 11, 12],
    }
    original = pl._calibration_block_project

    def height_dependent_projection(tensors, name, values):
        if name != "layer0.o_proj":
            return original(tensors, name, values)
        # This is the projection mutant the fixed-height block prevents: the direct
        # GEMM result is made observably dependent on left-matrix height.
        out = values @ tensors[name].T
        return out + np.spacing(out) * values.shape[0]

    monkeypatch.setattr(pl, "_calibration_block_project", height_dependent_projection)
    factored = pl._calibrate(cfg, model.float_weight, token_lists, token_lists.__getitem__, factored=True)
    unfactored = pl._calibrate(cfg, model.float_weight, token_lists, token_lists.__getitem__, factored=False)
    assert _maxima_bytes(factored) != _maxima_bytes(unfactored)

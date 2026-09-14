"""T-2693 slice-2 obligation (2): float prefix factoring preserves calibration bytes."""

import dataclasses
import struct

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

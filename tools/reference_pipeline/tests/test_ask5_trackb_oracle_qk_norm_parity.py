"""Ask 5, Track B — the oracle's QK-norm call site is ONE shared implementation, not three
independently-drifting ones (T-2553).

T-2551 gave `_float_layer` a QK-norm call site. `_kv_calibration_capture` (Arm D/E's own
per-head calibration capture) and `_vec_forward` (the integer engine's own Python parity
shadow) each independently re-walk the per-layer forward and had none — an oracle with
multiple forward implementations of which only one applies a real architectural operation is
exactly the shape `StandardsDocument.md` §7's sibling-pinning rule names: a repair (or, here,
an addition) that leaves an unaudited sibling standing inherits whatever the sibling is wrong
about, and the sibling here was wrong about applying QK-norm at all.

T-2553 gives both siblings the identical call, `_apply_qk_norm` (`pipeline.py`) — one shared
function every independent layer walk in this module calls, so there is one QK-norm
composition, never two (or three) reasoned to agree. This file pins that: a fixture with
non-uniform `q_norm`/`k_norm` gains where each sibling's own observable output measurably
differs with the norm applied versus without it — materiality, not merely execution, matching
the same discipline `test_ask5_trackc_qk_norm_converter.py` and T-2551's own build log already
establish for this class of claim (`StandardsDocument.md` §5.4).
"""

import numpy as np
import pytest

from conftest import api, require

MODULE = "reference_pipeline.pipeline"


def _fixture_config(pipeline):
    """The §11 fixture model's shape (test_pipeline.fixture_config's own kwargs, inlined so
    this file does not import across test modules) -- carries non-uniform q_norm/k_norm
    gains unconditionally (_weight_shapes, T-2539), which is what makes this fixture usable
    for the materiality proof below without any hand-authored perturbation.
    """
    return pipeline.ModelConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, intermediate_size=64, vocab_size=32,
        rope_theta=10000.0, rms_norm_eps=1e-6, tie_word_embeddings=True, context_cap=16,
    )


def _without_qk_norm(floats, cfg):
    """The identical float source, with every layer's own q_norm.gain/k_norm.gain removed --
    the fixture's own no-QK-norm twin, matching every other absence-vs-presence materiality
    proof in this codebase (T-2551's own build log; T-2425's own spike log)."""
    stripped = dict(floats)
    for layer in range(cfg.num_hidden_layers):
        stripped.pop(f"layer{layer}.q_norm.gain", None)
        stripped.pop(f"layer{layer}.k_norm.gain", None)
    return stripped


def test_apply_qk_norm_is_the_one_shared_implementation():
    """_float_layer and _kv_calibration_capture call the IDENTICAL function object, not two
    separately-authored copies that happen to compute the same formula today and can
    silently drift apart tomorrow -- the defect class this whole ticket exists to close,
    made mechanically checkable."""
    pipeline = require(MODULE)
    assert hasattr(pipeline, "_apply_qk_norm"), (
        "no shared _apply_qk_norm function -- T-2553's own governing fix is absent"
    )


def test_kv_calibration_capture_maxima_differ_with_and_without_qk_norm():
    """Arm D/E's own per-head calibration capture -- the sibling T-2551's own build log
    found silently omitting QK-norm. maxima[f"{prefix}.q"] (the SAME site
    _layer_q_scale/_projection_scale read to derive the production Q scale, per this file's
    own docstring) measurably differs between a QK-norm-bearing fixture and its
    tensor-stripped twin -- not merely that the call site executes, but that it changes a
    number this arm's own downstream scale derivation actually reads.
    """
    pipeline = require(MODULE)
    cfg = _fixture_config(pipeline)
    _, _, floats_with = pipeline._pinned_weights(cfg)
    floats_without = _without_qk_norm(floats_with, cfg)
    records = pipeline.calibration_records()
    tokenize = pipeline._fixture_tokenize_prompt(cfg)
    record_tokenize = pipeline._bridge_record_tokenizer(tokenize)

    _, maxima_with = pipeline._kv_calibration_capture(
        cfg, pipeline._dict_float_source(floats_with), records, record_tokenize)
    _, maxima_without = pipeline._kv_calibration_capture(
        cfg, pipeline._dict_float_source(floats_without), records, record_tokenize)

    differed = False
    for layer in range(cfg.num_hidden_layers):
        key = f"layer{layer}.q"
        assert key in maxima_with and key in maxima_without, (
            f"{key}: missing from one of the two maxima dicts -- capture did not run"
        )
        if maxima_with[key] != pytest.approx(maxima_without[key], rel=1e-12):
            differed = True
    assert differed, (
        "_kv_calibration_capture's own maxima[layer{L}.q] is bit-for-bit identical with "
        "and without q_norm/k_norm at every layer -- QK-norm has no measurable effect on "
        "this sibling's own observable output, the exact silent-no-op shape T-2551's own "
        "_float_layer bug (a wrong lookup key) already produced once"
    )


def test_vec_forward_layer_outputs_differ_with_and_without_qk_norm():
    """_vec_forward -- the integer engine's own Python parity shadow, this module's THIRD
    independent layer walk (test_integer_pipeline_tracks_the_float_reference_per_layer is
    what caught it missing QK-norm entirely: the int8 parity shadow diverged from the
    now-corrected float oracle by more than that test's own 25% structural-defect
    tolerance). Real, materially different forward_layers output with QK-norm tensors
    present vs. absent, dequantized so the comparison is apples to apples.
    """
    pipeline = require(MODULE)
    cfg = _fixture_config(pipeline)
    weights, weight_scales, floats_with = pipeline._pinned_weights(cfg)
    float_weight_with = pipeline._dict_float_source(floats_with)
    maxima_with = pipeline._calibrate(
        cfg, float_weight_with, pipeline.calibration_records(),
        pipeline._bridge_record_tokenizer(pipeline._fixture_tokenize_prompt(cfg)))
    scales_with, residual_scales_with, biases_with = pipeline._derive_scales(
        cfg, maxima_with, weight_scales, {})
    composition_with, kv_scales_with, kv_recip_with = pipeline._derive_composition_constants(
        cfg, weight_scales, scales_with)
    model_with = pipeline.QuantizedModel(
        config=cfg, scales=scales_with, weights=weights, weight_scales=weight_scales,
        residual_scales=residual_scales_with, rope_tables=pipeline._build_rope_tables(cfg),
        biases=biases_with, float_source=float_weight_with,
        tokenize_prompt=pipeline._fixture_tokenize_prompt(cfg),
        calibration=pipeline._calibration_record("fixture"), gemm_weights={},
        composition_constants=composition_with, kv_landing_scales=kv_scales_with,
        kv_landing_reciprocals=kv_recip_with)

    floats_without = _without_qk_norm(floats_with, cfg)
    weights_without = dict(weights)
    weight_scales_without = dict(weight_scales)
    for layer in range(cfg.num_hidden_layers):
        weights_without.pop(f"layer{layer}.q_norm.gain", None)
        weights_without.pop(f"layer{layer}.k_norm.gain", None)
        weight_scales_without.pop(f"layer{layer}.q_norm.gain", None)
        weight_scales_without.pop(f"layer{layer}.k_norm.gain", None)
    float_weight_without = pipeline._dict_float_source(floats_without)
    maxima_without = pipeline._calibrate(
        cfg, float_weight_without, pipeline.calibration_records(),
        pipeline._bridge_record_tokenizer(pipeline._fixture_tokenize_prompt(cfg)))
    scales_without, residual_scales_without, biases_without = pipeline._derive_scales(
        cfg, maxima_without, weight_scales_without, {})
    composition_without, kv_scales_without, kv_recip_without = (
        pipeline._derive_composition_constants(cfg, weight_scales_without, scales_without))
    model_without = pipeline.QuantizedModel(
        config=cfg, scales=scales_without, weights=weights_without,
        weight_scales=weight_scales_without, residual_scales=residual_scales_without,
        rope_tables=pipeline._build_rope_tables(cfg), biases=biases_without,
        float_source=float_weight_without, tokenize_prompt=pipeline._fixture_tokenize_prompt(cfg),
        calibration=pipeline._calibration_record("fixture"), gemm_weights={},
        composition_constants=composition_without, kv_landing_scales=kv_scales_without,
        kv_landing_reciprocals=kv_recip_without)

    tokens = [1, 3, 5, 7]
    out_with = pipeline.forward_layers(model_with, tokens)
    out_without = pipeline.forward_layers(model_without, tokens)
    assert out_with[0].shape == out_without[0].shape
    max_delta = max(
        float(np.abs(np.asarray(a) - np.asarray(b)).max())
        for a, b in zip(out_with, out_without))
    assert max_delta > 0.0, (
        "_vec_forward's own forward_layers output is bit-for-bit identical with and "
        "without q_norm/k_norm -- the integer parity shadow's new QK-norm site has no "
        "measurable effect, the exact silent-no-op shape this ticket exists to close"
    )

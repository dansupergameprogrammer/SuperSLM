"""Curie's red suite for sslm_convert_validate.py (S-HARDEN-3, F13, T-164 SS3.3).

Builds a tiny synthetic calibrated model directly (a `types.SimpleNamespace`
satisfying the same attribute contract `build_sections` reads) rather than
importing convert_model.py or reference_pipeline -- convert_model.py's own
`_fold_ops_tensor`/`_ctx_fold_tensor` defaults reach `reference_pipeline`
(vendored in-tree at `tools/reference_pipeline/`, T-2123/T-2137; no longer a
cross-tree `D:\\Wizard\\Tools` import as of B5) only through a LAZY import,
triggered by those default functions actually being called -- which this
suite avoids entirely by supplying its own fold functions, keeping every test
below dependency-light and independent of whether `transformers`/`tokenizers`
are installed, with numpy the only dependency this file itself needs.

Each rejection test asserts the SPECIFIC `.code` that fired -- the
S-HARDEN-2 review's lesson, restated in this slot's own brief: "a rejection
test must assert WHICH rejection fired ... or a shadowing check makes it a
vacuous witness."
"""

import types

import numpy as np
import pytest

import sslm_convert_validate as V


# --- a tiny, geometrically coherent synthetic calibrated model -------------
# hidden_size = num_attention_heads * head_dim = 2 * 4 = 8. One layer, one
# weight tensor, one KV head (kv_heads=1, heads=2 -> group=2, divisible).

def _cfg(**overrides):
    fields = dict(hidden_size=8, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                  head_dim=4, intermediate_size=8, vocab_size=4, context_cap=4)
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


class _Scales:
    """Stands in for the spike's `model.scales` -- `.scale(key)` returns a
    positive float per named site, injectable per test."""

    def __init__(self, table):
        self._table = table

    def scale(self, key):
        return self._table[key]


def _valid_model(**cfg_overrides):
    cfg = _cfg(**cfg_overrides)
    return types.SimpleNamespace(
        config=cfg,
        weights={"layer0.w": np.array([[-128, 0, 127], [1, 2, 3]], dtype=np.int8)},
        dynamic_biases={"layer0.site": (30, np.array([1, 2, 3], dtype=np.int64))},
        rope_tables=(np.array([100, 200], dtype=np.int64), np.array([-100, -200], dtype=np.int64)),
        weight_scales={"layer0.w": [0.5, 1.5]},
        scales=_Scales({"layer0.v_head0.scale": 0.75}),
        composition_constants={"scale": (1000000, 0)},
        kv_landing_scales={"kv": (1, 0)},
        kv_landing_reciprocals={"kv": (1, 0, 0)},
    )


def _identity_fold_ops_tensor(channel_scales):
    """A trivial stand-in for convert_model.py's real `_fold_ops_tensor`
    (which calls the cross-tree spike's `pipeline._reference_fold`): every
    channel folds to the pass-through identity row. Sufficient for exercising
    validate_model's OWN logic, which treats the fold function as an
    injected pure function it does not implement."""
    return np.asarray([(1, 0, 0) for _ in channel_scales], dtype=np.int32)


def _identity_ctx_fold_tensor(model, layer):
    cfg = model.config
    return np.asarray([(1, 0, 0) for _ in range(cfg.num_attention_heads)], dtype=np.int32)


def _validate(model, **kwargs):
    kwargs.setdefault("fold_ops_tensor", _identity_fold_ops_tensor)
    kwargs.setdefault("ctx_fold_tensor", _identity_ctx_fold_tensor)
    kwargs.setdefault("running_unicode_version", "15.1.0")
    V.validate_model(model, **kwargs)


# --- the valid fixture itself must pass every check ------------------------

def test_valid_synthetic_model_passes():
    _validate(_valid_model())


# --- F13's own reproduction: int16 [128,-129] silently becomes int8
#     [-128,127] under the OLD unguarded `np.asarray(..., dtype=np.int8)`
#     path -- reproduced here directly (not via convert_model.py, which this
#     module cannot import in CI) as the red-first population proof, then
#     validate_model on the SAME hostile array is asserted to reject it. ---

def test_old_unguarded_cast_silently_wraps_reproducing_f13():
    hostile = np.array([128, -129], dtype=np.int16)
    wrapped = np.asarray(hostile, dtype=np.int8)  # the pre-S-HARDEN-3 writer's own line
    assert list(wrapped) == [-128, 127], (
        "this assertion documents the F13 defect's exact shape (NumPy 2.5.1's own coercion "
        "behaviour) -- if it ever fails, NumPy's int16->int8 cast stopped silently wrapping "
        "and this reproduction is stale, not that the defect is fixed"
    )


def test_rejects_weights_out_of_int8_range_reproducing_f13():
    model = _valid_model()
    model.weights["layer0.w"] = np.array([[128, -129, 0], [1, 2, 3]], dtype=np.int16)
    with pytest.raises(V.ConverterValidationError) as exc:
        _validate(model)
    assert exc.value.code == "WeightsOutOfInt8Range", exc.value.code


def test_rejects_weights_non_integral_value():
    model = _valid_model()
    model.weights["layer0.w"] = np.array([[1.5, 0.0, 1.0], [1.0, 2.0, 3.0]], dtype=np.float64)
    with pytest.raises(V.ConverterValidationError) as exc:
        _validate(model)
    assert exc.value.code == "WeightsOutOfInt8RangeNotIntegral", exc.value.code


# --- config geometry, incl. the zero-boundary (S3.3) ------------------------

def test_rejects_zero_attention_heads():
    model = _valid_model(num_attention_heads=0)
    with pytest.raises(V.ConverterValidationError) as exc:
        _validate(model)
    assert exc.value.code == "ZeroAttentionHeads", exc.value.code


def test_rejects_zero_key_value_heads_before_modulus_faults():
    # The coverage audit's own specification: `heads % kv_heads` would raise
    # ZeroDivisionError if reached with kv_heads == 0. pytest.raises pins the
    # exact exception TYPE (ConverterValidationError, never ZeroDivisionError)
    # as well as the code, so this cell fails loudly if the ordering regresses.
    model = _valid_model(num_key_value_heads=0)
    with pytest.raises(V.ConverterValidationError) as exc:
        _validate(model)
    assert exc.value.code == "ZeroKeyValueHeads", exc.value.code


def test_rejects_kv_heads_exceeds_heads():
    model = _valid_model(num_attention_heads=1, num_key_value_heads=2, head_dim=8, hidden_size=8)
    with pytest.raises(V.ConverterValidationError) as exc:
        _validate(model)
    assert exc.value.code == "KvHeadsExceedsHeads", exc.value.code


def test_rejects_heads_not_divisible_by_kv():
    model = _valid_model(num_attention_heads=3, num_key_value_heads=2, head_dim=4, hidden_size=12)
    with pytest.raises(V.ConverterValidationError) as exc:
        _validate(model)
    assert exc.value.code == "HeadsNotDivisibleByKv", exc.value.code


def test_rejects_hidden_size_geometry_mismatch():
    model = _valid_model(hidden_size=9)  # heads(2) * head_dim(4) = 8 != 9
    with pytest.raises(V.ConverterValidationError) as exc:
        _validate(model)
    assert exc.value.code == "HiddenSizeGeometryMismatch", exc.value.code


def test_accepts_mha_shape_kv_heads_equals_heads():
    model = _valid_model(num_attention_heads=2, num_key_value_heads=2, head_dim=4, hidden_size=8)
    model.scales = _Scales({"layer0.v_head0.scale": 0.5, "layer0.v_head1.scale": 0.5})
    _validate(model)  # must not raise


# --- scale positivity and fold bounds ---------------------------------------

def test_rejects_non_positive_weight_scale():
    model = _valid_model()
    model.weight_scales["layer0.w"] = [0.5, -1.0]
    with pytest.raises(V.ConverterValidationError) as exc:
        _validate(model)
    assert exc.value.code == "NonPositiveScale", exc.value.code


def test_rejects_non_finite_weight_scale():
    model = _valid_model()
    model.weight_scales["layer0.w"] = [0.5, float("nan")]
    with pytest.raises(V.ConverterValidationError) as exc:
        _validate(model)
    assert exc.value.code == "NonFiniteScale", exc.value.code


def test_rejects_non_positive_v_head_scale():
    model = _valid_model()
    model.scales = _Scales({"layer0.v_head0.scale": 0.0})
    with pytest.raises(V.ConverterValidationError) as exc:
        _validate(model)
    assert exc.value.code == "NonPositiveScale", exc.value.code


def test_rejects_fold_shift_out_of_bounds():
    model = _valid_model()

    def hostile_fold(channel_scales):
        return np.asarray([(0, 1, 32) for _ in channel_scales], dtype=np.int32)  # shift=32 -- one past 31

    with pytest.raises(V.ConverterValidationError) as exc:
        _validate(model, fold_ops_tensor=hostile_fold)
    assert exc.value.code == "FoldShiftOutOfBounds", exc.value.code


def test_rejects_fold_identity_not_bool():
    model = _valid_model()

    def hostile_fold(channel_scales):
        return np.asarray([(2, 1, 0) for _ in channel_scales], dtype=np.int32)  # identity=2

    with pytest.raises(V.ConverterValidationError) as exc:
        _validate(model, fold_ops_tensor=hostile_fold)
    assert exc.value.code == "FoldIdentityNotBool", exc.value.code


# --- required key sets --------------------------------------------------

def test_rejects_empty_required_group():
    model = _valid_model()
    model.dynamic_biases = {}
    with pytest.raises(V.ConverterValidationError) as exc:
        _validate(model)
    assert exc.value.code == "EmptyRequiredGroup", exc.value.code


def test_rejects_orphaned_weight_scale_key():
    model = _valid_model()
    model.weight_scales["layer0.ghost"] = [1.0]
    with pytest.raises(V.ConverterValidationError) as exc:
        _validate(model)
    assert exc.value.code == "OrphanedWeightScaleKey", exc.value.code


# --- rope table runtime domain and composition-constant no-UB floor --------

def test_rejects_rope_table_outside_runtime_domain():
    model = _valid_model()
    model.rope_tables = (np.array([1 << 30, (1 << 30) + 1], dtype=np.int64),
                         np.array([0, 0], dtype=np.int64))
    with pytest.raises(V.ConverterValidationError) as exc:
        _validate(model)
    assert exc.value.code == "RopeTablesOutOfRuntimeDomain", exc.value.code


def test_rejects_composition_scale_m_out_of_domain():
    model = _valid_model()
    model.composition_constants = {"scale": (-2147483648, 0)}  # one past the no-UB floor
    with pytest.raises(V.ConverterValidationError) as exc:
        _validate(model)
    assert exc.value.code == "CompositionScaleMOutOfDomain", exc.value.code


def test_rejects_composition_scale_e_out_of_domain():
    model = _valid_model()
    model.composition_constants = {"scale": (0, 8)}  # one past the upper no-UB floor
    with pytest.raises(V.ConverterValidationError) as exc:
        _validate(model)
    assert exc.value.code == "CompositionScaleEOutOfDomain", exc.value.code


# --- Unicode/version coherence ---------------------------------------------

def test_rejects_unicode_version_mismatch():
    model = _valid_model()
    with pytest.raises(V.ConverterValidationError) as exc:
        _validate(model, running_unicode_version="15.0.0")
    assert exc.value.code == "UnicodeVersionMismatch", exc.value.code


def test_accepts_matching_unicode_version():
    model = _valid_model()
    _validate(model, running_unicode_version="15.1.0")  # must not raise

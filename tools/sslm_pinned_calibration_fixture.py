"""sslm_pinned_calibration_fixture.py — the small pinned calibrated fixture
S-HARDEN-3's gate requires (SuperSLM_Plan.md §19: "the manifest emitted and
its invariant ranges gated in CI against a small pinned calibrated fixture;
a deliberately out-of-range calibration input rejected rather than wrapped").

A tiny, fully synthetic, deterministic in-memory model -- not a real
checkpoint (convert_model.py's real input requires the cross-tree spike and
a calibrated artifact directory unavailable in CI, the same constraint
sslm_convert_validate.py's module docstring documents). Every value is fixed
by construction (no randomness), so the fixture is hermetic per S-HARDEN-5's
own discipline: nothing here is generated from a source outside this
repository, and CI regenerates it fresh on every run rather than trusting a
committed binary.

Two entry points:
  - `build_valid_fixture()` -- passes every validate_model check and every
    invariant range this module's own gate asserts.
  - `build_out_of_range_fixture()` -- identical except one weight value is
    set to 200, one int16 past int8's [-128,127] ceiling: the exact F13
    reproduction shape (a value that WOULD silently wrap under the old
    `np.asarray(..., dtype=np.int8)` writer). validate_model must reject it.
"""

import types

import numpy as np


def _cfg():
    return types.SimpleNamespace(
        hidden_size=8, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=4, intermediate_size=8, vocab_size=4, context_cap=4, tie_word_embeddings=False,
        rope_theta=10000.0, rms_norm_eps=1e-6)


class _Scales:
    def __init__(self, table):
        self._table = table

    def scale(self, key):
        return self._table[key]


def _fold_ops_tensor(channel_scales):
    """Deterministic pass-through fold for every channel -- sufficient for a
    fixture whose purpose is exercising validate/serialize/verify, not
    calibration-accuracy behavior."""
    return np.asarray([(1, 0, 0) for _ in channel_scales], dtype=np.int32)


def _ctx_fold_tensor(model, layer):
    cfg = model.config
    return np.asarray([(1, 0, 0) for _ in range(cfg.num_attention_heads)], dtype=np.int32)


def _base_model():
    cfg = _cfg()
    weights = np.array([[-128, -1, 0, 1, 2, 3, 4, 127],
                        [5, -5, 10, -10, 20, -20, 30, -30]], dtype=np.int8)
    weight_scales = {"layer0.w": [0.5, 1.5]}
    # T-408 §4/§8 step 8: a planted float_weight dict alongside the existing planted
    # weights/weight_scales -- the same "every value fixed by construction" discipline
    # this module's own docstring states. Exact dequantization for every element except
    # one deliberately nonzero delta (row 1, last column: error_in_lsb = |0.6| / 1.5 =
    # 0.4), mirroring how build_out_of_range_fixture already plants a hostile value to
    # prove rejection fires -- this plants a genuine, nonzero quantization_error_percentiles
    # population instead.
    scale_row = np.asarray(weight_scales["layer0.w"], dtype=np.float64).reshape(-1, 1)
    dequantized = weights.astype(np.float64) * scale_row
    float_weights = dequantized.copy()
    float_weights[1, 7] += 0.6
    return types.SimpleNamespace(
        config=cfg,
        weights={"layer0.w": weights},
        dynamic_biases={"layer0.site": (30, np.array([1, 2, 3, 4], dtype=np.int64))},
        rope_tables=(np.array([1000, 2000, 3000], dtype=np.int64),
                    np.array([-1000, -2000, -3000], dtype=np.int64)),
        weight_scales=weight_scales,
        scales=_Scales({"layer0.v_head0.scale": 0.75}),
        composition_constants={"scale": (1000000, 0)},
        kv_landing_scales={"kv": (1, 0)},
        kv_landing_reciprocals={"kv": (1, 0, 0)},
        # A plain callable, not a bound method -- called identically to the real
        # QuantizedModel.float_weight as `model.float_weight(name)`.
        float_weight=lambda name: {"layer0.w": float_weights}[name],
    )


def build_valid_fixture():
    return _base_model()


def build_out_of_range_fixture():
    """F13's own reproduction, planted in the pinned fixture: one weight
    value at 200 -- representable in int16, one past int8's 127 ceiling, and
    the exact class NumPy's `astype(np.int8)` silently wraps to -56 (200 - 256)
    with no diagnostic under the pre-S-HARDEN-3 writer."""
    model = _base_model()
    hostile = model.weights["layer0.w"].astype(np.int16).copy()
    hostile[0, 0] = 200
    model.weights = {"layer0.w": hostile}
    return model


FOLD_OPS_TENSOR = _fold_ops_tensor
CTX_FOLD_TENSOR = _ctx_fold_tensor

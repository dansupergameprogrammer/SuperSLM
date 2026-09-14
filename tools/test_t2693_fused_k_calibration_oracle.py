"""Independent-vector checks for the T-2693 oracle."""

import struct

import fused_k_calibration_oracle as oracle


def test_composition_fused_sites_equal_the_independent_calibration_oracle():
    """The composition oracle's fused wide-RoPE/Q31 sites use this vector verbatim.

    This is deliberately an operator cell, rather than a second call to the
    forward implementation: the per-site values must agree with the independent
    calibration oracle on the same signed, Q30-domain input.
    """
    import composition_ref

    vector = _vector()
    expected = oracle.evaluate(vector)
    rotated = composition_ref._rope_rotate_wide(
        [expected["wide"]], [[vector["cos_q30"]]], [[vector["sin_q30"]]], 2)[0]
    assert rotated == expected["rotated"]
    score = composition_ref.rdbpot_oracle(
        sum(q * k * expected["ratio_q31"] for q, k in zip(vector["q_codes"], rotated)), 31)
    assert score == expected["score_q31"]


def _vector(**overrides):
    vector = {
        "gain_code": -128,
        "k_norm_m": 1 << 30,
        "k_norm_e": -30,
        "k_codes": [-1, 1],
        "q_codes": [127, -127],
        "cos_q30": 1 << 30,
        "sin_q30": 0,
        "landing_scale_bits": struct.unpack("<Q", struct.pack("<d", 127.0))[0],
        "norm_numerator": 1 << 30,
        "norm_denominator": 1 << 30,
    }
    vector.update(overrides)
    return vector


def test_full_loader_gain_domain_and_negative_floor_are_evaluated_not_clamped():
    result = oracle.evaluate(_vector())
    assert result["k_wide_source_scale"] == {"m": 2130706432, "e": -24}
    assert result["wide"] == [128, -128]
    assert result["rotated"] == [128, -128]
    assert result["ratio_q31"] == 1 << 31
    assert result["score_q31"] == 32512


def test_binary64_source_bits_are_the_only_landing_scale_authority():
    result = oracle.evaluate(_vector(k_norm_e=-37, landing_scale_bits=0x3FF0000000000000))
    assert result["binary64_bits"] == "3ff0000000000000"
    assert result["ratio_q31"] == 2130706432


def test_rejects_nonfinite_and_out_of_domain_inputs_loudly():
    try:
        oracle.evaluate(_vector(gain_code=128))
    except ValueError as exc:
        assert "Int8" in str(exc)
    else:
        raise AssertionError("out-of-domain gain was accepted")

    try:
        oracle.evaluate(_vector(landing_scale_bits=0x7FF0000000000000))
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("infinite binary64 scale was accepted")

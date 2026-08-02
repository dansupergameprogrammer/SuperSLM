"""Sanity checks for tools/logit_margin_report.py's own arithmetic (T-1681).

This is not a red-suite for the SLM engine -- T-1681 is a measurement tool,
not an engine change (contract: read-only on src/include; the one engine-
adjacent edit is tools/sslm_generate.cpp's new --dump-logits flag, which
writes an array the driver already computed and changes no forward-path
arithmetic). What this file protects is the measurement's OWN correctness:
the shared dump format both engines write to and this tool reads from, the
z-score normalisation the whole comparison depends on, and the per-position
analysis (rank, margins, Spearman) that produces the numbers the T-1681
report is trusted on. A silent bug in any of these would misreport the
underlying question (quantization tie-break vs. defect) without any
red/green signal from the C++ suite, which has no oracle over this file at
all.

Run with: python -m pytest tests/test_t1681_logit_margin_report.py -v
"""

import struct
import sys
from pathlib import Path

import numpy as np

_TOOLS_DIR = str(Path(__file__).resolve().parent.parent / "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

import logit_margin_report as lmr  # noqa: E402


def test_dump_format_roundtrip_int32(tmp_path):
    rows = np.array([[10, -5, 300], [1, 2, 3]], dtype=np.int32)
    path = tmp_path / "roundtrip_int32.bin"
    with open(path, "wb") as f:
        f.write(struct.pack("<QQ", rows.shape[0], rows.shape[1]))
        f.write(rows.tobytes())
    loaded = lmr.load_dump(path, np.int32)
    assert loaded.shape == rows.shape
    assert np.array_equal(loaded, rows)


def test_dump_format_roundtrip_float32(tmp_path):
    rows = np.array([[1.5, -2.25, 0.0], [3.125, 4.0, -1.0]], dtype=np.float32)
    path = tmp_path / "roundtrip_float32.bin"
    with open(path, "wb") as f:
        f.write(struct.pack("<QQ", rows.shape[0], rows.shape[1]))
        f.write(rows.tobytes())
    loaded = lmr.load_dump(path, np.float32)
    assert loaded.shape == rows.shape
    assert np.allclose(loaded, rows)


def test_zscore_row_known_values():
    row = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    z, mean, std = lmr.zscore_row(row)
    assert abs(mean - 3.0) < 1e-9
    assert abs(std - np.sqrt(2.0)) < 1e-9
    # z-scored row has mean 0, std 1 by construction.
    assert abs(float(z.mean())) < 1e-9
    assert abs(float(z.std()) - 1.0) < 1e-9


def test_zscore_row_constant_row_returns_zero_not_nan():
    # A degenerate row (every logit identical) has std 0 -- dividing by it
    # would produce NaN/Inf silently propagating into every downstream
    # margin and rank figure. zscore_row must return an all-zero row instead.
    row = np.array([7.0, 7.0, 7.0])
    z, mean, std = lmr.zscore_row(row)
    assert std == 0.0
    assert np.array_equal(z, np.zeros(3))


def test_analyze_position_agreeing_case_has_zero_margin_and_rank_one():
    # int8's choice IS float's own top-1 -> margin-to-choice is exactly 0
    # and its rank in float's own ordering is 1, by construction.
    float_row = np.array([5.0, 1.0, 3.0, 2.0], dtype=np.float32)
    int8_row = np.array([50, 10, 30, 20], dtype=np.int32)
    result = lmr.analyze_position(float_row, int8_row, int8_token=0, position=0)
    assert result.agree is True
    assert result.float_token == 0
    assert result.float_margin_top1_to_int8_choice == 0.0
    assert result.z_margin_top1_to_int8_choice == 0.0
    assert result.int8_token_rank_in_float_order == 1


def test_analyze_position_flip_case_reports_correct_rank_and_margin():
    # float's order (descending) is index 0 (5.0) > 2 (3.0) > 3 (2.0) > 1 (1.0).
    # int8 chooses index 3, float's THIRD-ranked token.
    float_row = np.array([5.0, 1.0, 3.0, 2.0], dtype=np.float32)
    int8_row = np.array([10, 40, 20, 50], dtype=np.int32)  # int8 itself prefers index 3
    result = lmr.analyze_position(float_row, int8_row, int8_token=3, position=0)
    assert result.agree is False
    assert result.float_token == 0
    assert result.int8_token == 3
    assert result.int8_token_rank_in_float_order == 3
    assert abs(result.float_margin_top1_to_int8_choice - (5.0 - 2.0)) < 1e-9
    assert abs(result.float_top1_top2_margin - (5.0 - 3.0)) < 1e-9


def test_analyze_position_rank_ties_broken_toward_lowest_index_stable_order():
    # Two float logits tied for top place (indices 0 and 1, both 5.0); the
    # stable descending argsort used by analyze_position keeps encounter
    # order among ties, so index 0 ranks ahead of index 1.
    float_row = np.array([5.0, 5.0, 1.0], dtype=np.float32)
    int8_row = np.array([1, 1, 1], dtype=np.int32)
    result = lmr.analyze_position(float_row, int8_row, int8_token=1, position=0)
    assert result.float_token == 0  # lowest-index tie winner
    assert result.int8_token_rank_in_float_order == 2  # index 1 ranks second, not first


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))

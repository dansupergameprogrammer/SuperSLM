"""Focused structural-vitality checks for the T-2703 smoke harness."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


_SMOKE_PATH = Path(__file__).with_name("t2703_real_artifact_smoke.py")
_SPEC = importlib.util.spec_from_file_location("t2703_real_artifact_smoke", _SMOKE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
smoke = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(smoke)


def test_nonfinite_and_zero_final_hidden_vectors_are_still_rejected():
    with pytest.raises(ValueError, match="zero or non-finite"):
        smoke._normalize(np.array([0.0, 0.0]))
    with pytest.raises(ValueError, match="zero or non-finite"):
        smoke._normalize(np.array([np.nan, 1.0]))


def test_structural_markers_remain_vital_without_quality_pass_bars():
    smoke._assert_structural_smoke("OK", [True, True], [True, True])
    with pytest.raises(AssertionError, match="verifier"):
        smoke._assert_structural_smoke("accepted", [True], [True])
    with pytest.raises(AssertionError, match="attention"):
        smoke._assert_structural_smoke("OK", [False], [True])
    with pytest.raises(AssertionError, match="byte-identity"):
        smoke._assert_structural_smoke("OK", [True], [False])


def test_forward_hash_parser_requires_every_byte_identity_field():
    complete = " ".join(f"{field}={'a' * 64}" for field in smoke._FORWARD_HASH_FIELDS)
    assert smoke._forward_hashes(complete) == {field: "a" * 64 for field in smoke._FORWARD_HASH_FIELDS}
    with pytest.raises(AssertionError, match="complete forward"):
        smoke._forward_hashes("status=" + "a" * 64)


def test_harness_source_has_no_fitted_quality_threshold_arguments():
    text = _SMOKE_PATH.read_text(encoding="utf-8")
    assert "--min-cosine" not in text
    assert "--min-rank-agreement" not in text
    assert '"quality_grading": "T-2704 Section 5 release gate"' in text

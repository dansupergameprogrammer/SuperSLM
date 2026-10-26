"""The artifact verifier's SchemaMasks check (array S2 finding; tests/array-s2/verify_schema_masks_check.py).

Collected with the other converter-v2 attach tests. The verifier under test is CV2_VERIFIER when set, else
the one the attach command finds under build/ (`cmake --build build --target sslm_verify`)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

import cv2_fixtures as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "array-s2"))
import verify_schema_masks_check as V  # noqa: E402


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    if not (F.TOKENIZER_DIR / "tokenizer.json").is_file():
        pytest.skip(f"pinned tokenizer absent at {F.TOKENIZER_DIR} (set T3094_TOKENIZER_DIR)")
    verifier = Path(os.environ["CV2_VERIFIER"]) if os.environ.get("CV2_VERIFIER") else V.default_verifier()
    return {r["case"]: r for r in V.run(verifier, tmp_path_factory.mktemp("verify_scm1"))}


def test_attach_with_the_verifier_under_test_succeeds(results):
    assert results["attach"]["ok"], results["attach"]


@pytest.mark.parametrize("case", [c for c, accept, _ in V.CASES if accept])
def test_verifier_accepts(results, case):
    assert results[case]["ok"], results[case]


@pytest.mark.parametrize("case", [c for c, accept, _ in V.CASES if not accept])
def test_verifier_refuses_with_schema_masks_rejected(results, case):
    assert results[case]["ok"], results[case]

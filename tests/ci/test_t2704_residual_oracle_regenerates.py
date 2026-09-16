"""T-2704 fixture lifecycle and independent-oracle checks."""
from __future__ import annotations

import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTS / "reference"))
import t2704_residual_oracle as oracle  # noqa: E402


def test_t2704_generator_reproduces_the_checked_in_fixture_byte_for_byte():
    assert oracle.OUT_PATH.read_text(encoding="ascii") == oracle.generate()


def test_t2704_normalization_endpoint_witnesses_are_independent_and_exact():
    assert oracle.normalize(1) == (1 << 30, 30)
    assert oracle.normalize(1 << 30) == (1 << 30, 0)
    assert oracle.normalize((1 << 31) - 1) == ((1 << 31) - 1, 0)
    assert oracle.normalize(1 << 31) == (1 << 30, -1)
    assert oracle.reciprocal(1 << 30) == 1 << 32


def test_t2704_half_away_and_m1_tie_construction_are_nonvacuous():
    assert oracle.round_half_away(3, 1) == 2
    assert oracle.round_half_away(-3, 1) == -2
    assert oracle.landing(-127, 1, 1 << 32, -17, -17, 30) == -127

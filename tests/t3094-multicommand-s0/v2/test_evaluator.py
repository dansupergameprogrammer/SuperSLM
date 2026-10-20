"""T-3103 S0 executed cell: the reference evaluator over TL-1 to TL-20 (plus extensions TL-13c and TL-21),
and each s4.5 mutant failing its own fixture after the healthy run passes."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference_evaluator as re_  # noqa: E402
from timeline_fixtures import FIXTURES, MUTANT_FIXTURES  # noqa: E402


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_healthy_reference_matches_hand_ledger(name):
    assert re_.evaluate(FIXTURES[name]) == FIXTURES[name]["expected"]


def test_every_plan_mutant_is_mapped():
    assert set(MUTANT_FIXTURES) == set(re_.MUTANTS) and len(re_.MUTANTS) == 14


@pytest.mark.parametrize("mutant", sorted(MUTANT_FIXTURES))
def test_mutant_fails_its_own_fixture(mutant):
    for name in MUTANT_FIXTURES[mutant]:
        assert re_.evaluate(FIXTURES[name], [mutant]) != FIXTURES[name]["expected"], f"{mutant} survives {name}"


def test_plan_fixtures_alone_do_not_kill_two_mutants():
    """MX-7: with only the plan's TL-1 to TL-20, these two mutants survive every fixture."""
    plan_only = {k: v for k, v in FIXTURES.items() if k not in ("TL-13c", "TL-21")}
    for mutant in ("scaled_as_float", "ignore_wait_completion"):
        assert all(re_.evaluate(f, [mutant]) == f["expected"] for f in plan_only.values()), mutant

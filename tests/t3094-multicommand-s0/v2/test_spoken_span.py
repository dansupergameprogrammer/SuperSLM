"""T-3103 spoken-span check cells, including GV-1 (a synonym added to a gold set must be refused)."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spoken_span as ss  # noqa: E402


def row(utterance, atom, spans, **extra):
    return {"row": "r1", "utterance": utterance,
            "document": {"intents": [{"action": "fire", "subject": "alpha", "target": "gate", "conditions": [atom],
                                      "depends_on_previous": False}]}, "spans": [[spans]], **extra}


def A(e, s, cmp="none", value="-"):
    return {"role": "start", "entity": e, "state": s, "cmp": cmp, "value": value, "negated": False, "watch": "auto"}


GOOD = row("Alpha, fire on the gate when the beacon is lit.", A("beacon", "lit"),
           {"entity": ["beacon", "the beacon"], "state": ["lit", "is lit"], "value": []})


def test_healthy_gold_passes():
    assert ss.check_row(GOOD) == []


def test_gv1_synonym_added_is_refused():
    bad = copy.deepcopy(GOOD)
    bad["spans"][0][0]["state"].append("burning")
    assert ss.check_row(bad) == [("r1", "intent0.atom0.state", "not-a-spoken-span")]


@pytest.mark.parametrize("utt,atom,spans", [
    ("Hold the ford until your health drops below half.", A("bravo", "health", "below", "half"),
     {"entity": ["bravo"], "state": ["health"], "value": ["half"]}),
])
def test_resolved_possessive_exemption(utt, atom, spans):
    r = row(utt, atom, spans, resolved=[[{"entity": True}]], addressee="bravo")
    assert ss.check_row(r) == []
    r2 = row(utt, atom, spans)
    assert ss.check_row(r2)                 # without the resolved flag it is not a spoken span


def test_numbers_percent_and_clitic():
    r = row("Fire once the tower's shield is under twenty percent.", A("tower shield", "shield", "below", "20%"),
            {"entity": ["tower's shield", "tower shield"], "state": ["shield"], "value": ["twenty percent", "20%"]})
    assert ss.check_row(r) == []
    # C maps number words only from zero to twenty: "30 percent" is not a span of "thirty percent".
    r2 = row("Fire once the tower's shield is under thirty percent.", A("tower shield", "shield", "below", "30%"),
             {"entity": ["tower shield"], "state": ["shield"], "value": ["30 percent"]})
    assert ss.check_row(r2) == [("r1", "intent0.atom0.value", "not-a-spoken-span")]


def test_sentinel_rules():
    bad = copy.deepcopy(GOOD)
    bad["spans"][0][0]["value"] = ["none"]
    assert ss.check_row(bad)[0][2] == "sentinel"
    num = row("Fire when fewer than none of them stand.", A("them", "stand", "at_most", "none"),
              {"entity": ["them"], "state": ["stand"], "value": ["-"]})
    assert ss.check_row(num)[0][2] == "sentinel-as-spoken-value"


def test_placeholder_atom_exempt():
    r = row("Move to the ridge.", {"role": "start", "entity": "no condition", "state": "no condition", "cmp": "none",
                                   "value": "-", "negated": False, "watch": "auto"},
            {"entity": ["no condition"], "state": ["no condition"], "value": []})
    assert ss.check_row(r) == []

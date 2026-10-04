"""T-3103 S0 contract cells: frozen IntentDoc v2 / G1 schemas, product predicate, s1.1 semantic rules
and canonicalizer C (plan rev 15 s1, s1.1, s5.1, s6 rows 2 and 7)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import canon  # noqa: E402
import contract as c  # noqa: E402

# Plan s1.1 worked examples X1-X3, byte for byte; the plan records 224, 438 and 532 bytes.
X1 = b'{"intents":[{"action":"fire","subject":"alpha wing","target":"north gate","conditions":[{"role":"start","entity":"beacon","state":"lit","cmp":"none","value":"-","negated":false,"watch":"auto"}],"depends_on_previous":false}]}'
X2 = b'{"intents":[{"action":"fire","subject":"bravo squad","target":"bridge","conditions":[{"role":"start","entity":"gate","state":"open","cmp":"none","value":"-","negated":true,"watch":"auto"}],"depends_on_previous":false},{"action":"hold","subject":"bravo squad","target":"bridge","conditions":[{"role":"stop","entity":"bravo squad","state":"health","cmp":"below","value":"half","negated":false,"watch":"auto"}],"depends_on_previous":false}]}'
X3 = b'{"intents":[{"action":"move","subject":"charlie team","target":"ridge","conditions":[],"depends_on_previous":false},{"action":"scan","subject":"charlie team","target":"ridge","conditions":[],"depends_on_previous":false},{"action":"fire","subject":"charlie team","target":"tower","conditions":[{"role":"start","entity":"beacon","state":"lit","cmp":"none","value":"-","negated":false,"watch":"auto"},{"role":"start","entity":"gate","state":"open","cmp":"none","value":"-","negated":false,"watch":"auto"}],"depends_on_previous":true}]}'


def ok(raw, mode="v2"):
    return c.product_predicate(raw, accepting=True, stop="Complete", mode=mode)


def refused(raw, mode="v2"):
    with pytest.raises(c.Refusal) as caught:
        ok(raw, mode)
    return caught.value.status


def test_worked_examples_accept_with_plan_byte_counts():
    assert [len(x) for x in (X1, X2, X3)] == [224, 438, 532]
    for x in (X1, X2, X3):
        doc = ok(x)
        assert c.canonical_bytes(doc) == x


def test_schema_files_match_the_contract():
    v2 = json.loads((HERE / "intentdoc-v2.schema.json").read_text(encoding="utf-8"))
    intent = v2["properties"]["intents"]["items"]
    assert v2["properties"]["intents"]["minItems"] == 1
    assert tuple(intent["required"]) == c.INTENT_KEYS and tuple(intent["properties"]) == c.INTENT_KEYS
    atom = intent["properties"]["conditions"]["items"]
    assert intent["properties"]["conditions"]["minItems"] == 0
    assert tuple(atom["required"]) == c.ATOM_KEYS and tuple(atom["properties"]) == c.ATOM_KEYS
    assert tuple(atom["properties"]["role"]["enum"]) == c.ROLES
    assert tuple(atom["properties"]["cmp"]["enum"]) == c.CMPS
    assert tuple(atom["properties"]["watch"]["enum"]) == c.WATCHES
    for node in (v2, intent, atom):
        assert node["additionalProperties"] is False
    g1 = json.loads((HERE / "g1.schema.json").read_text(encoding="utf-8"))
    assert tuple(g1["properties"]["intent"]["required"]) == c.G1_KEYS
    g1g = json.loads((HERE / "g1g.schema.json").read_text(encoding="utf-8"))
    assert tuple(g1g["properties"]["intent"]["required"]) == c.G1G_KEYS
    for s in (v2, g1, g1g):
        assert "maxItems" not in json.dumps(s)


# s5.1 production-path must-rejects (InvalidConditionAtom) and must-accepts
def atom_doc(**fields):
    atom = {"role": "start", "entity": "gate", "state": "open", "cmp": "none", "value": "-", "negated": False,
            "watch": "auto"}
    atom.update(fields)
    return json.dumps({"intents": [{"action": "fire", "subject": "a", "target": "b", "conditions": [atom],
                                    "depends_on_previous": False}]}, separators=(",", ":")).encode()


@pytest.mark.parametrize("fields", [dict(cmp="none", value="half"), dict(cmp="below", value="-"),
                                    dict(cmp="none", value="None"), dict(role="start_any"), dict(entity="-"),
                                    dict(state="-")])
def test_semantic_must_rejects(fields):
    assert refused(atom_doc(**fields)) == "InvalidConditionAtom"


@pytest.mark.parametrize("fields", [dict(cmp="at_most", value="none"), dict(cmp="at_least", value="all"),
                                    dict(cmp="none", value="-")])
def test_semantic_must_accepts(fields):
    ok(atom_doc(**fields))


def test_two_start_any_accept_and_rules_run_on_exact_bytes():
    doc = json.loads(atom_doc(role="start_any"))
    doc["intents"][0]["conditions"].append({**doc["intents"][0]["conditions"][0], "entity": "flare"})
    ok(json.dumps(doc, separators=(",", ":")).encode())
    assert refused(atom_doc(cmp="none", value=" -")) == "InvalidConditionAtom"


@pytest.mark.parametrize("raw,status", [
    (X1.replace(b'"subject":"alpha wing",', b""), "InvalidDocument"),
    (X1.replace(b'"action":"fire","subject":"alpha wing"', b'"subject":"alpha wing","action":"fire"'), "InvalidDocument"),
    (X1.replace(b'"watch":"auto"}', b'"watch":"auto","extra":1}'), "InvalidDocument"),
    (b'{"intents":[]}', "InvalidDocument"),
    (X1.replace(b'"state":"lit"', b'"state":", "'), "InvalidDocument"),
    (X1.replace(b'"action":"fire"', b'"action":"fire","action":"fire"'), "InvalidDocument"),
    (X1.replace(b'"conditions":[{', b'"conditions":[[').replace(b'"auto"}]', b'"auto"]]'), "InvalidDocument"),
    (X1[:-2], "InvalidDocument"),
    (X1.replace(b'"action":"fire"', b'"action":"attack"'), "InvalidDocument"),
])
def test_shape_refusals(raw, status):
    assert raw != X1
    assert refused(raw) == status


@pytest.mark.parametrize("accepting,stop,status", [(False, "Complete", "NonacceptingStop"), (True, "Budget", "BudgetStop"),
                                                   (True, "DeadEnd", "SchemaDeadEnd"), (True, "StopToken", "NonacceptingStop")])
def test_stop_classification(accepting, stop, status):
    with pytest.raises(c.Refusal) as caught:
        c.product_predicate(X1, accepting=accepting, stop=stop)
    assert caught.value.status == status


def test_g1_and_g1g_documents():
    g1 = b'{"intent":{"action":"fire","subject":"alpha wing","target":"north gate","condition":{"role":"start","entity":"beacon","state":"lit","cmp":"none","value":"-","negated":false,"watch":"auto"},"depends_on_previous":false}}'
    doc = ok(g1, "g1")
    assert c.g1_to_v2(doc) == ok(X1)
    ungated = json.dumps({"intent": {"action": "move", "subject": "a", "target": "b", "gated": False,
                                     "condition": c.PLACEHOLDER_ATOM, "depends_on_previous": False}},
                         separators=(",", ":")).encode()
    assert c.g1_to_v2(ok(ungated, "g1g"))["intents"][0]["conditions"] == []
    assert refused(g1.replace(b'"value":"-"', b'"value":""'), "g1") == "InvalidDocument"


# Canonicalizer C contract cells (plan s6 row 7)
@pytest.mark.parametrize("text,field,expected", [
    ("The Beacon", "entity", "beacon"), ("  a   north   gate. ", "entity", "north gate"),
    ("is lit", "state", "lit"), ("has been destroyed", "state", "destroyed"), ("gets hot", "state", "hot"),
    ("tower's shield", "entity", "tower shield"), ("tower’s shield", "entity", "tower shield"),
    ("three", "value", "3"), ("twenty percent", "value", "20 %"), ("30 per cent", "value", "30 %"),
    ("café", "entity", "café"), ("-", "value", "-"), ("-", "entity", "-"),
])
def test_canon_applies_each_listed_step(text, field, expected):
    assert canon.canon(text, field) == expected


@pytest.mark.parametrize("text,field,expected", [
    ("their turret", "entity", "their turret"), ("your health", "entity", "your health"),
    ("burning", "state", "burning"), ("lights", "state", "lights"), ("half", "value", "half"),
    ("twenty-one", "value", "twenty-one"), ("is lit", "entity", "is lit"), ("the the gate", "entity", "the gate"),
])
def test_canon_applies_nothing_else(text, field, expected):
    assert canon.canon(text, field) == expected

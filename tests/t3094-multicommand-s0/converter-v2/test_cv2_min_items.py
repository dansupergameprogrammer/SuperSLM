"""Item 2 — per-array minItems, its limit and diagnostics, the next-depth case, maxItems refusal
(plan §4.1 converter, resource admission and maxItems paragraphs; §6 rows 2, 4, 11; D-SLM8087)."""
from __future__ import annotations

import pytest

import cv2_fixtures as F
import cv2_oracle as B
import cv2_product as P


def _docs(schema, root, inner):
    return B.serialize(schema, F.doc(*F.intents(root, inner)))


@pytest.mark.parametrize("minimum", (1, 2, 3, 5))
def test_root_minimum_is_exact_and_loops(minimum):
    """Root minimum 5 is the largest IntentDoc v2 root minimum inside every release state limit (count
    model v2: 6 copies of the intent automaton reach 4,016 NFA states, 8 reach 5,350)."""
    schema = F.with_min(minimum, 0)
    mp = P.healthy_bytes(schema)
    for n in range(minimum, minimum + 3):
        assert P.accepts_raw_bytes(mp, _docs(schema, n, 0)), f"{n} intents refused at minItems {minimum}"
    if minimum > 1:
        assert not P.accepts_raw_bytes(mp, _docs(schema, minimum - 1, 0)), \
            f"{minimum - 1} intents accepted at minItems {minimum}: minimum truncated"
    assert not P.accepts_raw_bytes(mp, b'{"intents":[]}'), "root [] accepted"


@pytest.mark.parametrize("minimum", (0, 1, 2, 8))
def test_inner_minimum_is_exact_and_loops(minimum):
    schema = F.with_min(1, minimum)
    mp = P.healthy_bytes(schema)
    for n in range(minimum, minimum + 3):
        assert P.accepts_raw_bytes(mp, _docs(schema, 2, n)), f"{n} atoms refused at inner minItems {minimum}"
    if minimum:
        short = B.serialize(schema, F.doc(F.intent(conditions=F.atoms(minimum)), F.intent(conditions=F.atoms(minimum - 1))))
        assert not P.accepts_raw_bytes(mp, short), \
            f"an intent with {minimum - 1} atoms accepted at inner minItems {minimum}: inner minimum ignored or truncated"
    empty = B.serialize(schema, F.doc(F.intent()))
    assert P.accepts_raw_bytes(mp, empty) == (minimum == 0), "inner [] verdict wrong"


def test_inner_minimum_is_not_applied_from_the_root():
    """A converter that applies minItems only at the root (§6 row 4 mutant) treats both arrays alike;
    here the root needs 2 and the inner array 0, then the reverse."""
    a = F.with_min(2, 0)
    mp = P.healthy_bytes(a)
    assert P.accepts_raw_bytes(mp, _docs(a, 2, 0)), "root 2 / inner 0: two intents with [] refused"
    b = F.with_min(1, 2)
    mp = P.healthy_bytes(b)
    assert P.accepts_raw_bytes(mp, _docs(b, 1, 2)) and not P.accepts_raw_bytes(mp, _docs(b, 1, 1))


# ------------------------------------------------------------------ the minItems limit on IntentDoc v2
#
# Inner: at the release limit (8), conditions minItems 8 is admitted on every other guard and 9 crosses only
# minItems (count model v2, byte vocabulary). Root: IntentDoc v2 with intents minItems 8 already crosses the
# NFA and DFA state limits (5,350 / 4,982 at the byte vocabulary), so the root pair on the production schema
# runs under an injected minItems limit of 5 (5 admitted on every guard, 6 crosses only minItems). The
# release-limit root pair is the frozen synthetic R_MIN_ROOT (below and in test_cv2_resources).

def _min_limit_case(level):
    if level == "root":
        return F.with_min(5, 0), F.with_min(6, 0), {"minItems": 5}, (5, 0), (4, 0)
    return F.with_min(1, 8), F.with_min(1, 9), None, (1, 8), (1, 7)


def _compile(schema, limits):
    return P.compile_bytes_limited(schema, limits) if limits else P.compile_bytes(schema)


@pytest.mark.parametrize("level", ("root", "inner"))
def test_min_items_limit_under_admits_exact_minimum(level):
    under, _, limits, exact, short = _min_limit_case(level)
    P.healthy_bytes(F.INTENTDOC)
    mp = _compile(under, limits)
    assert P.accepts_raw_bytes(mp, _docs(under, *exact)), f"{level}: exact minimum at the limit refused"
    assert not P.accepts_raw_bytes(mp, _docs(under, *short)), f"{level}: one below the minimum accepted"


@pytest.mark.parametrize("level", ("root", "inner"))
def test_min_items_limit_over_refused_naming_its_array(level):
    """§4.1: the limit applies to every array independently and the diagnostic names the array; the other
    array stays in range. Healthy control first: the under schema compiles at the same limits."""
    under, over, limits, _, _ = _min_limit_case(level)
    P.healthy_bytes(F.INTENTDOC)
    _compile(under, limits)
    with pytest.raises(P.SchemaCompileError, match=F.diagnostic_regex("minItems")) as info:
        _compile(over, limits)
    message = str(info.value)
    assert not any(d in message for d in F.other_diagnostics("minItems"))
    if level == "root":
        assert F.names_pointer(message, F.ROOT_POINTERS) and "conditions" not in message, message
    else:
        assert F.names_pointer(message, F.INNER_POINTERS), message


@pytest.mark.parametrize("name", ("R_MIN_ROOT", "R_MIN_INNER"))
def test_frozen_min_pairs_at_byte_vocabulary(name):
    """The frozen S0 pairs (resource-pairs-v2.json): minItems is a schema property, so the byte vocabulary
    exercises the same guard; the pinned-vocabulary run is in test_cv2_resources."""
    p = F.pair(name)
    mp = P.healthy_bytes(p["under_schema"])
    inner = name == "R_MIN_INNER"
    def raw(n):
        return B.serialize(p["under_schema"], {"a": [{"b": ["x"] * n}]} if inner else {"a": ["x"] * n})
    assert P.accepts_raw_bytes(mp, raw(8)) and P.accepts_raw_bytes(mp, raw(9)) and not P.accepts_raw_bytes(mp, raw(7))
    with pytest.raises(P.SchemaCompileError, match=F.diagnostic_regex("minItems")):
        P.compile_bytes(p["over_schema"])


# ------------------------------------------------------------------ the next nesting depth

@pytest.mark.parametrize("family", ("intentdoc_value", "three_nested_arrays"))
def test_depth_limit_admits_eight_and_refuses_nine(family):
    if family == "intentdoc_value":
        under, over = F.deepen_value(2), F.deepen_value(3)
        value = F.doc(F.intent(conditions=[F.atom(cmp="below", value=F.deep_value(2))]))
    else:
        under, over = F.triple_array(), F.triple_array(True)
        value = {"a": [{"b": []}, {"b": [{"c": [{"d": "x"}, {"d": "x"}]}]}]}
    mp = P.healthy_bytes(under)
    assert P.accepts_raw_bytes(mp, B.serialize(under, value)), "depth-8 document refused"
    with pytest.raises(P.SchemaCompileError, match=F.diagnostic_regex("recursion_depth")) as info:
        P.compile_bytes(over)
    assert not any(d in str(info.value) for d in F.other_diagnostics("recursion_depth"))


# ------------------------------------------------------------------ maxItems refused at every array

@pytest.mark.parametrize("level,value", (("intents", 1), ("intents", 4), ("intents", 8),
                                         ("conditions", 0), ("conditions", 1), ("conditions", 4)))
def test_max_items_refused_by_name(level, value):
    """§4.1: the production converter rejects maxItems even where a small value could be flattened."""
    P.healthy_bytes(F.INTENTDOC)
    schema = F.fresh()
    F.node(schema, level)["maxItems"] = value
    with pytest.raises(P.SchemaCompileError, match="maxItems"):
        P.compile_bytes(schema)


@pytest.mark.parametrize("level", ("intents", "conditions"))
@pytest.mark.parametrize("defect,expected", (
    ("items_missing", "items"), ("items_tuple", "items"), ("items_boolean", "items"),
    ("min_negative", "minItems"), ("min_fraction", "minItems"), ("min_string", "minItems"), ("min_bool", "minItems"),
))
def test_malformed_array_named_refusal(level, defect, expected):
    P.healthy_bytes(F.INTENTDOC)
    schema = F.fresh()
    array = F.node(schema, level)
    if defect == "items_missing":
        del array["items"]
    elif defect == "items_tuple":
        array["items"] = [array["items"]]
    elif defect == "items_boolean":
        array["items"] = True
    else:
        array["minItems"] = {"min_negative": -1, "min_fraction": 1.5, "min_string": "1", "min_bool": True}[defect]
    with pytest.raises(P.SchemaCompileError, match=expected):
        P.compile_bytes(schema)


@pytest.mark.parametrize("location", ("intent", "atom"))
@pytest.mark.parametrize("defect,expected", (("optional_key", "required"), ("reordered_keys", "required"),
                                             ("open_object", "additionalProperties")))
def test_item_object_named_refusal(location, defect, expected):
    P.healthy_bytes(F.INTENTDOC)
    schema = F.fresh()
    item = F.node(schema, location)
    if defect == "optional_key":
        item["required"] = item["required"][:-1]
    elif defect == "reordered_keys":
        item["required"] = list(reversed(item["required"]))
    else:
        item["additionalProperties"] = True
    with pytest.raises(P.SchemaCompileError, match=expected):
        P.compile_bytes(schema)

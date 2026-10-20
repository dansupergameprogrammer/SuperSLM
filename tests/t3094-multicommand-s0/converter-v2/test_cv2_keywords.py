"""Item 3 — keyword-by-node matrix on the IntentDoc v2 nodes (plan §6 row 2; §8 S0a "independent keyword
catalogue at every new node"). Every unsupported placement gets a named compile refusal and no output;
every permitted annotation is accepted and leaves the language unchanged."""
from __future__ import annotations

import re

import pytest

import cv2_cases as C
import cv2_fixtures as F
import cv2_keywords as K
import cv2_product as P
from keyword_catalogue import VALUES


@pytest.mark.parametrize("location,keyword", K.REFUSALS, ids=[f"{l}-{k}" for l, k in K.REFUSALS])
def test_unsupported_keyword_refused_by_name_at_node(location, keyword):
    P.healthy_bytes(F.INTENTDOC)
    schema = F.fresh()
    F.node(schema, location)[keyword] = VALUES[keyword]
    with pytest.raises(P.SchemaCompileError, match=re.escape(keyword)):
        P.compile_bytes(schema)


@pytest.mark.parametrize("location,annotation", K.ANNOTATION_PLACEMENTS,
                         ids=[f"{l}-{a}" for l, a in K.ANNOTATION_PLACEMENTS])
def test_annotation_accepted_and_language_unchanged(location, annotation):
    baseline = P.healthy_bytes(F.INTENTDOC)
    schema = F.fresh()
    F.node(schema, location)[annotation] = K.ANNOTATION_VALUES.get(annotation, "metadata")
    mp = P.healthy_bytes(schema)
    for _, raw in C.ACCEPT[:4] + C.REJECT[:6]:
        assert P.accepts_raw_bytes(mp, raw) == P.accepts_raw_bytes(baseline, raw)
    assert mp.transitions == baseline.transitions and mp.accepting == baseline.accepting


@pytest.mark.parametrize("keyword", ("$schema", "$id"))
def test_root_identifiers_accepted(keyword):
    schema = F.fresh()
    schema[keyword] = VALUES[keyword]
    mp = P.healthy_bytes(schema)
    assert P.accepts_raw_bytes(mp, C.ONE)

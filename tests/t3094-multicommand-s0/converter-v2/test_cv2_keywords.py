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


CONTROL = "vendorUnknownConstraint"


def _occurrences(keyword, message):
    """Occurrences of `keyword` as a whole token: not inside a longer identifier (S0a fold 6)."""
    return len(re.findall(r"(?<![A-Za-z0-9_$])" + re.escape(keyword) + r"(?![A-Za-z0-9_])", message))


def _refusal(location, keyword):
    schema = F.fresh()
    F.node(schema, location)[keyword] = VALUES.get(keyword, 1)
    with pytest.raises(P.SchemaCompileError) as info:
        P.compile_bytes(schema)
    return str(info.value)


@pytest.mark.parametrize("location,keyword", K.REFUSALS, ids=[f"{l}-{k}" for l, k in K.REFUSALS])
def test_unsupported_keyword_refused_by_name_at_node(location, keyword):
    """§6 row 2: a named compile refusal. The keyword must occur as a whole token, and more often than in the
    refusal of a different unsupported keyword at the same node, so a refusal phrased "... not supported at
    ..." does not name `not` by accident."""
    P.healthy_bytes(F.INTENTDOC)
    message = _refusal(location, keyword)
    control = _refusal(location, CONTROL) if keyword != CONTROL else ""
    assert _occurrences(keyword, message) > _occurrences(keyword, control), \
        f"refusal does not name {keyword!r}: {message!r} (control: {control!r})"


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

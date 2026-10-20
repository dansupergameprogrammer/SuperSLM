"""Item 4 — the seven resource limits (plan §4.1 resource admission contract, re-derived limits, "How these
are confirmed", the fixture-pair table and the memory paragraph; §6 rows 4 and 11; D-SLM8087, D-SLM8095,
D-SLM8104, D-SLM8120).

Three groups of product cells:
- must-admit: IntentDoc v2 and G1 compile at the release limits on the pinned vocabulary, report all seven
  counts, and reconcile with count model v2;
- release pairs: each frozen pair from resource-pairs-v2.json that count model v2 isolates at the release
  limits (R_MIN_ROOT, R_MIN_INNER, R_DEPTH, R_NFA, R_TRANS, R_SECTION) runs at the release limits on the
  pinned vocabulary; R_DFA and R_MEM are injection-only (limits-freeze.json, D-SLM8120) and have no cell here;
- injection pairs: one IntentDoc v2 pair per guard (cv2_injection), the target limit injected through the
  production guard, every other limit at release.
A refusal is credited only when it names its own guard and no other guard's diagnostic.
"""
from __future__ import annotations

import pytest

import count_model_v2 as cm
import cv2_fixtures as F
import cv2_injection as I
import cv2_oracle as B
import cv2_product as P

RED_COUNTS = "red-unimplemented: the compiled artifact reports no resource_counts"
RED_LIMITS = "red-unimplemented: the converter takes no limits argument"
EXACT = ("minItems", "recursion_depth", "dfa_states", "token_transitions", "serialized_mask_bytes")


def _counts(mp):
    counts = getattr(mp, "resource_counts", None)
    assert counts is not None, RED_COUNTS
    assert set(F.LIMIT_KEYS) <= set(counts), f"resource_counts lacks {sorted(set(F.LIMIT_KEYS) - set(counts))}"
    return counts


def _refused_by(key, fn):
    with pytest.raises(P.SchemaCompileError, match=F.diagnostic_regex(key)) as info:
        fn()
    others = [d for d in F.other_diagnostics(key) if d in str(info.value)]
    assert not others, f"{key} refusal also names {others}: a different guard's refusal earns no credit"


@pytest.mark.parametrize("name,schema,section", (("IntentDoc v2", F.INTENTDOC, "intents"), ("G1", F.G1, "intent_gate")),
                         ids=("intentdoc_v2", "g1"))
def test_must_admit_at_release_limits_on_pinned_vocabulary(name, schema, section, real_vocab):
    """§4.1 (3): compiles under the release limits; minItems at each array, depth, DFA states, transitions
    and section bytes equal count model v2 exactly; NFA states and tracked bytes are the converter's own
    construction and allocation accounting, recorded and held under their limits (see the record)."""
    vocab, pieces = real_vocab
    mp = P.healthy_real(schema, vocab)
    counts = _counts(mp)
    model = cm.count(schema, section, pieces, F.REAL_VOCAB_SIZE)
    expected = F.model_counts_to_contract(model)
    mins = counts["minItems"]
    assert isinstance(mins, dict) and sorted(mins.values()) == sorted(model["minItems"].values()), mins
    for key in EXACT[1:]:
        assert counts[key] == expected[key], f"{name} {key}: converter {counts[key]} vs count model v2 {expected[key]}"
    for key in F.LIMIT_KEYS[1:]:
        assert counts[key] <= F.RELEASE[key], f"{name} {key} {counts[key]} over release {F.RELEASE[key]}"
    assert len(mp.transitions) == model["dfa_states"]
    assert sum(map(len, mp.transitions.values())) == model["token_transitions"]


def test_must_admit_counts_at_byte_vocabulary():
    """Fast reconciliation of the same counts at the byte vocabulary (one cell for the builder's loop)."""
    mp = P.healthy_bytes(F.INTENTDOC)
    counts = _counts(mp)
    model = I.counts(F.INTENTDOC)
    for key in EXACT[1:]:
        assert counts[key] == model[key], f"{key}: converter {counts[key]} vs count model v2 {model[key]}"


RELEASE_PAIRS = ("R_MIN_ROOT", "R_MIN_INNER", "R_DEPTH", "R_NFA", "R_TRANS", "R_SECTION")


def _example(schema):
    def ex(node):
        if "enum" in node:
            return node["enum"][0]
        kind = node["type"]
        if kind == "boolean":
            return False
        if kind == "string":
            return "go"
        if kind == "object":
            return {k: ex(v) for k, v in node["properties"].items()}
        return [ex(node["items"]) for _ in range(max(node.get("minItems", 0), 1))]
    return ex(schema)


@pytest.mark.parametrize("name", RELEASE_PAIRS)
def test_release_pair_on_pinned_vocabulary(name, real_vocab):
    """Under: compiles at the release limits and accepts its exact required content (for minItems pairs,
    exactly 8 items accepted and 7 refused). Over: that limit's named refusal, with no artifact."""
    vocab, _ = real_vocab
    p = F.pair(name)
    key = F.PAIR_TARGET[name]
    mp = P.healthy_real(p["under_schema"], vocab)
    raw = B.serialize(p["under_schema"], _example(p["under_schema"]))
    assert P.accepts_ids(mp, vocab.bytewise(raw))
    if key == "minItems":
        inner = name == "R_MIN_INNER"
        def doc(n):
            return B.serialize(p["under_schema"], {"a": [{"b": ["x"] * n}]} if inner else {"a": ["x"] * n})
        assert P.accepts_ids(mp, vocab.bytewise(doc(8))) and not P.accepts_ids(mp, vocab.bytewise(doc(7)))
    _refused_by(key, lambda: P.compile_schema_to_mask_pages(p["over_schema"], vocab.pieces,
                                                             special_ids=vocab.special))


@pytest.mark.parametrize("name", tuple(I.PAIRS))
def test_injected_limit_pair_isolates_its_guard(name):
    """The target limit injected through the production guard; under admitted and walks a document, over
    refused by its own diagnostic. Healthy control first (IntentDoc v2 compiles)."""
    P.healthy_bytes(F.INTENTDOC)
    under, over, key, limit = I.pair(name)
    limits = {key: limit}
    try:
        mp = P.compile_bytes(under, limits=limits)
    except TypeError as error:
        pytest.fail(f"{RED_LIMITS}: {error}")
    assert P.accepts_raw_bytes(mp, B.serialize(under, _example(under)))
    _refused_by(key, lambda: P.compile_bytes(over, limits=limits))


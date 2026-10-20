"""Reference controls for the converter-v2 red suite. Green by design: they establish that the independent
oracles, fixtures and counts are what the product cells assume, before any product verdict is read."""
from __future__ import annotations

import hashlib
import json

import pytest

import cv2_cases as C
import cv2_fixtures as F
import cv2_oracle as B
import cv2_tokens as T
import contract as A  # v2/contract.py: Oracle A (strict parse plus frozen-shape check)
import count_model_v2 as cm
from keyword_catalogue import KEYWORDS, VALUES


# ------------------------------------------------------------------ Oracle B against hand verdicts and Oracle A

@pytest.mark.parametrize("case,raw", C.ACCEPT, ids=[c[0] for c in C.ACCEPT])
def test_oracle_b_accepts_and_oracle_a_agrees(case, raw):
    B.parse(F.INTENTDOC, raw)
    A.shape(A.parse(raw), "v2")
    B.check_pieces(F.INTENTDOC, raw, [bytes([b]) for b in raw])


@pytest.mark.parametrize("case,raw", C.REJECT, ids=[c[0] for c in C.REJECT])
def test_oracle_b_rejects(case, raw):
    with pytest.raises(B.Reject):
        B.parse(F.INTENTDOC, raw)


@pytest.mark.parametrize("case", ("whitespace_after_colon", "whitespace_between_items", "trailing_newline",
                                  "leading_space", "whitespace_in_empty_conditions"))
def test_a_accepted_b_rejected_counterexamples(case):
    """§7: whitespace variants are valid JSON of the right shape (Oracle A accepts) that the canonical
    language refuses (Oracle B); they are explicit counterexamples, not healthy-compiler failures."""
    raw = dict(C.REJECT)[case]
    A.shape(A.parse(raw), "v2")
    with pytest.raises(B.Reject):
        B.parse(F.INTENTDOC, raw)


def test_oracle_b_minimum_per_array():
    for root, inner in ((1, 0), (2, 0), (1, 2), (3, 3)):
        schema = F.with_min(root, inner)
        ok = B.serialize(schema, F.doc(*F.intents(root, inner)))
        B.parse(schema, ok)
        if root > 1:
            with pytest.raises(B.Reject, match=r"\$\.intents has"):
                B.parse(schema, B.serialize(schema, F.doc(*F.intents(root - 1, inner))))
        if inner:
            with pytest.raises(B.Reject, match=r"conditions has"):
                B.parse(schema, B.serialize(schema, F.doc(*F.intents(root, inner - 1))))


@pytest.mark.parametrize("case", T.CASES, ids=[c[0] for c in T.CASES])
def test_oracle_b_token_boundary_cases(case):
    _, raw, offset, piece, verdict = case
    B.parse(F.INTENTDOC, raw)
    pieces = T.pieces(raw, offset, piece)
    if verdict:
        B.check_pieces(F.INTENTDOC, raw, pieces)
    else:
        with pytest.raises(B.Reject, match="compound opening"):
            B.check_pieces(F.INTENTDOC, raw, pieces)


def test_real_vocabulary_boundary_tokens_exist_and_are_ordinary(real_vocab):
    vocab, _ = real_vocab
    assert len(vocab.special) == 22 and vocab.special == frozenset(range(151643, 151665))
    for piece in T.COMPOUNDS + tuple(bytes([b]) for b in range(256)):
        tid = vocab.id_of(piece)
        assert tid not in vocab.special and vocab.pieces[tid] == piece


@pytest.mark.parametrize("case", T.CASES, ids=[c[0] for c in T.CASES])
def test_oracle_b_real_vocabulary_token_ids(case, real_vocab):
    vocab, _ = real_vocab
    _, raw, offset, piece, verdict = case
    ids = vocab.ids(T.pieces(raw, offset, piece))
    if verdict:
        B.check_tokens(F.INTENTDOC, ids, vocab)
    else:
        with pytest.raises(B.Reject, match="compound opening"):
            B.check_tokens(F.INTENTDOC, ids, vocab)
    B.check_tokens(F.INTENTDOC, vocab.bytewise(raw), vocab)


def test_oracle_b_refuses_special_and_duplicate_ids(real_vocab):
    vocab, _ = real_vocab
    ids = vocab.bytewise(C.ONE)
    with pytest.raises(B.Reject, match="special id"):
        B.check_tokens(F.INTENTDOC, ids[:5] + [151643] + ids[5:], vocab)
    shadow = B.Vocabulary(list(vocab.pieces[:256]) + [b"{"], [])
    with pytest.raises(B.Reject, match="not the lowest id"):
        B.check_tokens(F.INTENTDOC, [256] + shadow.bytewise(C.ONE)[1:], shadow)


# ------------------------------------------------------------------ count model v2 and the frozen limits

PLAN_REV10 = {  # plan §4.1 rev-10 table: limits and measured counts (IntentDoc v2, G1)
    "limits": {"minItems": 8, "recursion_depth": 8, "nfa_states": 4096, "dfa_states": 4096,
               "token_transitions": 8388608, "section_bytes": 134217728, "tracked_bytes": 268435456},
    "IntentDoc v2": {"recursion_depth": 6, "nfa_states": 681, "dfa_states": 635, "token_transitions": 3074805,
                     "section_bytes": 36660995, "tracked_bytes": 86095411},
    "G1": {"recursion_depth": 4, "nfa_states": 424, "dfa_states": 399, "token_transitions": 1921739,
           "section_bytes": 22953415, "tracked_bytes": 53852695},
}


def test_frozen_limits_equal_plan_rev10_table():
    assert F.LIMITS_FREEZE["limits"] == PLAN_REV10["limits"]
    for name in ("IntentDoc v2", "G1"):
        assert F.LIMITS_FREEZE["counts_count_model_v2"][name] == PLAN_REV10[name]
    assert F.LIMITS_FREEZE["isolation"]["R_MEM"] == "injection-only"
    assert F.LIMITS_FREEZE["isolation"]["R_DFA"] == "injection-only"


@pytest.mark.parametrize("name,schema,section", (("IntentDoc v2", F.INTENTDOC, "intents"), ("G1", F.G1, "intent_gate")),
                         ids=("intentdoc_v2", "g1"))
def test_count_model_v2_reproduces_must_admit_counts_at_pinned_vocabulary(name, schema, section, real_vocab):
    _, pieces = real_vocab
    counts = cm.count(schema, section, pieces, F.REAL_VOCAB_SIZE)
    assert {k: counts[k] for k in PLAN_REV10[name]} == PLAN_REV10[name]
    assert counts["minItems"] == ({"$.intents": 1, "$.intents[].conditions": 0} if name == "IntentDoc v2" else {})
    assert not cm.crossings(counts)


def test_resource_pairs_file_is_the_frozen_s0_record():
    raw = (F.V2 / "resource-pairs-v2.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == RESOURCE_PAIRS_SHA256
    assert F.RESOURCE_PAIRS["limits"] == PLAN_REV10["limits"]
    assert [p["pair"] for p in F.RESOURCE_PAIRS["pairs"]] == [
        "R_MIN_ROOT", "R_MIN_INNER", "R_DEPTH", "R_NFA", "R_DFA", "R_TRANS", "R_SECTION", "R_MEM"]


RESOURCE_PAIRS_SHA256 = "4990fd53fb791a2c786ebf10556dd1c9aa6e1deced257e9baf22e3b82f637e74"
FROZEN_SHA256 = {
    "limits-freeze.json": "b5b78ca578cc6d06715cf88ea10c2b614e9e059ba99344e98483529a0b362384",
    "intentdoc-v2.schema.json": "bc51375c20bf7c0ba3179cc64f6af0a8272766790153b8c90b2a2dfa0d3d03cc",
    "g1.schema.json": "e6fc1d68064462d319b91a5864b2496b7b6993ab37257e2a8c34388df6bbcb29",
}


@pytest.mark.parametrize("name", tuple(FROZEN_SHA256))
def test_frozen_inputs_unchanged(name):
    assert hashlib.sha256((F.V2 / name).read_bytes()).hexdigest() == FROZEN_SHA256[name]


@pytest.mark.parametrize("name", ("R_MIN_ROOT", "R_MIN_INNER", "R_DEPTH", "R_NFA", "R_DFA", "R_TRANS", "R_SECTION"))
def test_release_pair_recounts_and_isolation_at_pinned_vocabulary(name, real_vocab):
    """Recount each frozen pair with count model v2 (never the converter): the under schema crosses no
    release limit, the over schema crosses only its target, or the pair is declared injection-only."""
    _, pieces = real_vocab
    p = F.pair(name)
    under = cm.count(p["under_schema"], "pair", pieces, F.REAL_VOCAB_SIZE)
    over = cm.count(p["over_schema"], "pair", pieces, F.REAL_VOCAB_SIZE)
    assert under == p["under"] and over == p["over"]
    assert cm.crossings(under) == []
    target = F.FREEZE_KEY[F.PAIR_TARGET[name]]
    if name == "R_DFA":
        assert p["injection_only"] and cm.crossings(over) == ["nfa_states", "dfa_states"]
    else:
        assert cm.crossings(over) == [target] and p["isolated"]


def test_stored_pair_schemas_out_of_required_order_are_exactly_r_section():
    """Finding (routed in the record): sort_keys re-sorted R_SECTION's stored properties; every other stored
    pair schema is in `required` order. F.pair() serves the counted order."""
    assert F.stored_order_defects() == [("R_SECTION", "under_schema"), ("R_SECTION", "over_schema")]


def test_r_mem_is_declared_injection_only_with_its_reason():
    p = F.pair("R_MEM")
    assert p["injection_only"] and not p["isolated"] and "D-SLM8120" in p["note"]


# ------------------------------------------------------------------ the injection pairs (byte vocabulary)

import cv2_injection as I  # noqa: E402


@pytest.mark.parametrize("name", tuple(I.PAIRS))
def test_injection_pair_isolates_its_target_under_count_model_v2(name):
    """Each pair crosses only its target at the injected limits, counted by count model v2 at the byte
    vocabulary; the size pairs leave a factor of at least 1.5 on each side of the injected limit, so the
    converter's own count may differ from the model's within that factor without changing the verdict."""
    under, over, key, limit = I.pair(name)
    cu, co = I.counts(under), I.counts(over)
    limits = I.limits(key, limit)
    assert I.crossing(cu, limits) == [] and I.crossing(co, limits) == [key]
    if key not in ("minItems", "recursion_depth"):
        assert limit >= 1.5 * cu[key] and co[key] >= 1.5 * limit


# ------------------------------------------------------------------ keyword catalogue on the v2 nodes

def test_keyword_catalogue_covers_every_v2_node():
    assert KEYWORDS == set(VALUES)
    assert len(F.NODES) == 16
    assert {kind for _, kind in F.NODES.values()} == {"object", "array", "enum", "string", "boolean"}
    for location in F.NODES:
        assert F.allowed(location) <= KEYWORDS
        assert F.node(F.INTENTDOC, location) is not None
    import cv2_keywords as K
    assert len(K.REFUSALS) >= 800 and len(K.EXCLUDED) == 5


# ------------------------------------------------------------------ fixture schemas

def test_depth_fixtures_have_the_stated_depth():
    assert cm.depth(F.INTENTDOC) == 6 and cm.depth(F.G1) == 4
    assert cm.depth(F.deepen_value(2)) == 8 and cm.depth(F.deepen_value(3)) == 9
    assert cm.depth(F.triple_array()) == 8 and cm.depth(F.triple_array(True)) == 9


def test_byte_vocabulary_counts_of_the_fixture_family_admit():
    for schema in (F.INTENTDOC, F.G1, F.with_min(5, 0), F.with_min(1, 8), F.deepen_value(2), F.triple_array()):
        assert I.crossing(I.counts(schema), dict(F.RELEASE)) == []
    # Finding: the production schema's root minItems pair is not isolatable at the release limits.
    assert I.crossing(I.counts(F.with_min(8, 0)), dict(F.RELEASE)) == ["nfa_states", "dfa_states"]
    assert I.crossing(I.counts(F.with_min(6, 0)), I.limits("minItems", 5)) == ["minItems"]
    assert I.crossing(I.counts(F.with_min(1, 9)), dict(F.RELEASE)) == ["minItems"]
    assert I.crossing(I.counts(F.deepen_value(3)), dict(F.RELEASE)) == ["recursion_depth"]
    assert I.crossing(I.counts(F.triple_array(True)), dict(F.RELEASE)) == ["recursion_depth"]

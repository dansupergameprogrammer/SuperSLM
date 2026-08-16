"""Suite for the T-066 spike harness's schema-constrained decoder.

Module under test: ``reference_pipeline.constrain``. Authored red-first, pinning the
contract before the module existed; the module has since landed (vendored with the
criterion-2 closure, T-1520) and every cell here runs green against it.

Sources realized here:

- ``Claude/Plans/SuperSLM_Plan.md`` §9 (constrained decoding: token-level DFA
  compiled over the artifact's actual vocabulary, per-state valid-token masks,
  masks applied to int32 logits before argmax; the non-empty-valid-set
  obligation placed on the compiler, not the runtime).
- ``Claude/Plans/SuperSLM_Plan.md`` §6.5 (greedy-only, argmax over int32 logits,
  pinned tie-break = lowest token index, no softmax).
- ``Claude/Plans/SuperSLM_Plan.md`` §17 dim 10 (the feature oracle:
  schema-constrained decode yields schema-VALID output — parse the produced
  tokens against the schema, not "same tokens twice"), §17 G-7 (DFA
  completeness), §17 G-7a (the non-empty-valid-set failure cell + the negative
  control).
- ``Claude/Docs/Shopkeeper_IntentExtraction_Schema.md`` §3 (the schema compiled
  here) and §5 (hard-constraint violations must be 0 by construction).

Cell map — see ``Claude/Curie/superslm-t066-constrained-decode-test-design-2026-07-14.md``:

    C1  conformance (dim 10 feature oracle)
    C2  G-7 completeness (+ the soundness cells that keep completeness from
        being satisfiable by an accept-everything DFA)
    C3  G-7a compile-time rejection, its paired satisfiable control, the
        positive per-state obligation, and the negative control
    C4  greedy argmax + pinned tie-break (§6.5)
    C5  tokenizer-boundary correctness (§9)
    C6  forced-singleton states (the jump-forward precondition that is in the
        spike's scope; jump-forward itself is routed to the S5 runtime slot —
        see the record)
    C7  a real tokenizer vocabulary (skipped cleanly when transformers or a
        cached tokenizer is unavailable)

The DFA/mask/argmax logic is exercised against a small fake vocabulary and
hand-built int32 logit vectors; no model, torch, or transformers import is
required by any cell except C7.

Every test in this file is expected to FAIL until ``reference_pipeline.constrain``
exists. The failure reason is reported explicitly by ``_constrain()``.
"""

from __future__ import annotations

import importlib
import json
import pathlib
from typing import Any, Mapping, Sequence

import pytest

# --------------------------------------------------------------------------
# The module under test — imported per-test so a missing implementation fails
# each cell with a stated reason rather than erroring out collection.
# --------------------------------------------------------------------------

_CONSTRAIN_MODULE = "reference_pipeline.constrain"


def _constrain():
    """Import the module under test, or fail the calling test as red-unimplemented."""
    try:
        return importlib.import_module(_CONSTRAIN_MODULE)
    except ImportError as exc:  # pragma: no cover - the red state
        pytest.fail(
            f"red-unimplemented: {_CONSTRAIN_MODULE} is not importable ({exc}). "
            "This suite pins its contract; the implementation is Hastings's."
        )


# --------------------------------------------------------------------------
# The schema — Shopkeeper_IntentExtraction_Schema.md §3, transcribed as the
# compilable subset §9 names (objects with known keys, enums, bool; no optional
# keys, no recursion).
# --------------------------------------------------------------------------

INTENT_VALUES = (
    "book",
    "reschedule",
    "cancel",
    "price_query",
    "availability_query",
    "design_query",
    "complaint",
    "out_of_domain",
)

SLOT_VALUES: dict[str, tuple[str, ...]] = {
    "artist": ("unspecified", "rin", "mox", "vale", "any"),
    "day": ("unspecified", "mon", "tue", "wed", "thu", "fri", "sat", "sun"),
    "time_block": ("unspecified", "morning", "afternoon", "evening"),
    "size": ("unspecified", "small", "medium", "large", "sleeve"),
    "placement": ("unspecified", "arm", "leg", "back", "chest", "hand", "neck"),
    "style": ("unspecified", "blackwork", "traditional", "realism", "script", "geometric"),
    "budget_band": ("unspecified", "under_200", "200_500", "500_1000", "over_1000"),
    "deposit_ok": ("unspecified", "yes", "no"),
}

POLARITY_VALUES = ("affirm", "negate")

# §3's F-5 field. `polarity: negate` alone could not say WHICH slot a negation attaches to —
# "Friday works, just not with Mox" affirms `day` and negates `artist`. `negated_slot` names the
# target and is `none` whenever `polarity == affirm`.
#
# Declared explicitly, NOT derived from SLOT_VALUES.keys(): the enum is `none` plus the
# POLARITY-NEUTRAL slots, which is not the same set as every slot. `deposit_ok` is not a member —
# its own enum already carries polarity (`yes | no`), so a deposit refusal is `deposit_ok=no` with
# `polarity=affirm`, and a negation target for it would mean "not not-a-deposit". Deriving this
# from the slot list would silently re-add it.
NEGATED_SLOT_VALUES = (
    "none",
    "artist",
    "day",
    "time_block",
    "size",
    "placement",
    "style",
    "budget_band",
)

SLOT_ORDER = tuple(SLOT_VALUES.keys())
DOC_ORDER = ("intent", "slots", "polarity", "negated_slot", "correction")

SHOPKEEPER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {"enum": list(INTENT_VALUES)},
        "slots": {
            "type": "object",
            "properties": {name: {"enum": list(values)} for name, values in SLOT_VALUES.items()},
            "required": list(SLOT_ORDER),
            "additionalProperties": False,
        },
        "polarity": {"enum": list(POLARITY_VALUES)},
        "negated_slot": {"enum": list(NEGATED_SLOT_VALUES)},
        "correction": {"type": "boolean"},
    },
    "required": list(DOC_ORDER),
    "additionalProperties": False,
}


# --------------------------------------------------------------------------
# The independent oracle — §3 restated as a validator this suite owns.
#
# The dim-10 feature oracle is only a feature oracle if the thing that judges
# the decoder's output is not the decoder. This validator never calls the module
# under test.
# --------------------------------------------------------------------------


def assert_schema_valid(doc: Any) -> None:
    """Assert ``doc`` satisfies Shopkeeper_IntentExtraction_Schema.md §3."""
    assert isinstance(doc, dict), f"document is not an object: {doc!r}"
    assert tuple(doc.keys()) == DOC_ORDER, f"document keys are {tuple(doc.keys())!r}, expected {DOC_ORDER!r}"

    assert doc["intent"] in INTENT_VALUES, f"intent {doc['intent']!r} is not a §3 enum value"

    slots = doc["slots"]
    assert isinstance(slots, dict), f"slots is not an object: {slots!r}"
    assert tuple(slots.keys()) == SLOT_ORDER, f"slot keys are {tuple(slots.keys())!r}, expected {SLOT_ORDER!r}"
    for name, values in SLOT_VALUES.items():
        assert slots[name] in values, f"slot {name}={slots[name]!r} is not a §3 enum value"

    assert doc["polarity"] in POLARITY_VALUES, f"polarity {doc['polarity']!r} is not a §3 enum value"
    assert doc["negated_slot"] in NEGATED_SLOT_VALUES, f"negated_slot {doc['negated_slot']!r} is not a §3 enum value"
    assert isinstance(doc["correction"], bool), f"correction is not a bool: {doc['correction']!r}"

    # DELIBERATELY NOT ASSERTED: §3's F-5 rule that `negated_slot` is `none` whenever
    # `polarity` is `affirm`. That is a CROSS-FIELD dependency, and §9's compilable subset
    # ("objects with known keys, enums, bounded strings/numbers/arrays") carries no construct
    # that can express one — so the schema compiler is never told the rule and cannot enforce it.
    #
    # This validator is the oracle for what the CONSTRAINT ENGINE must guarantee, so asserting the
    # IFF here would fail the decoder for a rule it was never given. The IFF is a corpus/semantic
    # property, gated by `test_corpus_invariants.py` and the adequacy gate, not by the mask.
    #
    # The consequence is a real gap and is routed to the planner as F-C6: schema doc §5 says
    # hard-constraint violations "must be 0 by construction", but a §3 rule now exists that the
    # §9 engine structurally cannot enforce. See the test-design record.


# --------------------------------------------------------------------------
# Canonical serialization.
#
# PINNED CONTRACT: the compiled DFA's language is the canonical serialization —
# schema key order, no insignificant whitespace, `true`/`false` for the bool.
# §9 does not state a serialization convention; that is a finding routed to the
# planner (see the record). This suite pins the convention it asserts against so
# the gap is visible rather than assumed.
# --------------------------------------------------------------------------


def canonical(doc: Mapping[str, Any]) -> str:
    ordered = {
        "intent": doc["intent"],
        "slots": {name: doc["slots"][name] for name in SLOT_ORDER},
        "polarity": doc["polarity"],
        "negated_slot": doc["negated_slot"],
        "correction": doc["correction"],
    }
    return json.dumps(ordered, separators=(",", ":"), ensure_ascii=False)


def make_doc(**overrides: Any) -> dict[str, Any]:
    """A §3-valid baseline document with the named fields overridden.

    Slot overrides are passed by slot name; ``intent``/``polarity``/``negated_slot``/
    ``correction`` by field name.

    Callers are responsible for keeping `polarity` and `negated_slot` coherent with §3's F-5
    rule. The fixtures below do, deliberately: every document this suite asserts the DFA must
    ACCEPT is valid under BOTH readings of §3 — the enum-only reading the compiler is given, and
    the stricter reading that includes the prose IFF. That way the completeness sweep cannot
    accidentally pin the engine's inability to enforce the IFF (F-C6) as a requirement that it
    must not.
    """
    doc: dict[str, Any] = {
        "intent": "book",
        "slots": {name: "unspecified" for name in SLOT_ORDER},
        "polarity": "affirm",
        "negated_slot": "none",
        "correction": False,
    }
    for key, value in overrides.items():
        if key in SLOT_VALUES:
            doc["slots"][key] = value
        else:
            doc[key] = value
    return doc


# --------------------------------------------------------------------------
# The fake vocabulary + a greedy longest-match tokenizer.
#
# Deliberately built so that the canonical serialization of a §3 document is
# genuinely multi-token per value (`geometric` -> `geo` + `metric`) and contains
# tokens that SPAN a value/punctuation boundary (`","`, `":"`), which is the
# retokenization hazard §9 answers by compiling over token sequences. The C5
# fixtures assert both properties hold before testing against them, so a vocab
# change cannot silently make those cells vacuous.
# --------------------------------------------------------------------------

JUNK_TOKEN = "\x00"  # appears in no canonical serialization; pinned at id 0.

_MERGES = (
    '{"',
    '":"',
    '","',
    '":',
    '":{"',
    '"}',
    "unspec",
    "ified",
    "geo",
    "metric",
    "black",
    "work",
    "true",
    "false",
    "under",
    "_200",
    "book",
    "morning",
    "arm",
)


def _sweep_documents() -> list[dict[str, Any]]:
    docs = [make_doc()]
    for value in INTENT_VALUES:
        docs.append(make_doc(intent=value))
    for name, values in SLOT_VALUES.items():
        for value in values:
            docs.append(make_doc(**{name: value}))
    # `polarity` and `negated_slot` are swept together, never independently: §3's F-5 rule ties
    # them, so a document with `polarity=affirm, negated_slot=artist` is valid to the compiler but
    # contradicts §3's prose. Sweeping them as coherent pairs keeps every fixture valid under both
    # readings — see make_doc.
    docs.append(make_doc(polarity="affirm", negated_slot="none"))
    docs.append(make_doc(polarity="negate", negated_slot="artist", artist="mox"))
    for value in NEGATED_SLOT_VALUES:
        if value == "none":
            docs.append(make_doc(polarity="affirm", negated_slot="none"))
        else:
            # The negated slot is also given a specified value, matching how the corpus labels a
            # negation ("not Tuesday" → day=tue, negated_slot=day).
            specified = next(v for v in SLOT_VALUES[value] if v != "unspecified")
            docs.append(make_doc(polarity="negate", negated_slot=value, **{value: specified}))
    for value in (True, False):
        docs.append(make_doc(correction=value))
    return docs


SWEEP_DOCUMENTS = _sweep_documents()


def _build_vocab() -> tuple[str, ...]:
    chars: set[str] = set()
    for doc in SWEEP_DOCUMENTS:
        chars |= set(canonical(doc))
    for merge in _MERGES:
        chars |= set(merge)
    return (JUNK_TOKEN,) + tuple(sorted(chars)) + tuple(sorted(_MERGES))


VOCAB = _build_vocab()
VOCAB_SIZE = len(VOCAB)
JUNK_ID = VOCAB.index(JUNK_TOKEN)
assert JUNK_ID == 0, "the junk token must sit at the lowest index for the C3 negative control"


def tokenize(text: str) -> list[int]:
    """Greedy longest-match tokenization over VOCAB. Deterministic by construction."""
    by_length = sorted(range(VOCAB_SIZE), key=lambda i: (-len(VOCAB[i]), i))
    out: list[int] = []
    pos = 0
    while pos < len(text):
        for token_id in by_length:
            piece = VOCAB[token_id]
            if piece and text.startswith(piece, pos):
                out.append(token_id)
                pos += len(piece)
                break
        else:  # pragma: no cover - a vocab bug, not a test outcome
            raise AssertionError(f"fixture vocabulary cannot tokenize at offset {pos} of {text!r}")
    return out


def detokenize(tokens: Sequence[int]) -> str:
    return "".join(VOCAB[t] for t in tokens)


# --------------------------------------------------------------------------
# int32 logits
# --------------------------------------------------------------------------

INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1


def logits(**by_id: int) -> list[int]:
    """A zero int32 logit vector with the named token ids set."""
    vector = [0] * VOCAB_SIZE
    for token_id, value in by_id.items():
        vector[int(token_id)] = value
    return vector


# --------------------------------------------------------------------------
# DFA walking helpers — this suite's own reachability, so the completeness and
# per-state cells do not lean on the implementation's notion of "reachable".
# --------------------------------------------------------------------------


def reachable_states(dfa) -> set[int]:
    seen = {dfa.start}
    stack = [dfa.start]
    while stack:
        state = stack.pop()
        for token_id in dfa.valid_tokens(state):
            nxt = dfa.step(state, token_id)
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def drive(dfa, tokens: Sequence[int]) -> int:
    """Drive ``tokens`` from the start state; returns the final state."""
    state = dfa.start
    for index, token_id in enumerate(tokens):
        valid = dfa.valid_tokens(state)
        assert token_id in valid, (
            f"token {token_id} ({VOCAB[token_id]!r}) at position {index} is not in the valid set "
            f"of state {state} — the DFA rejects a string the schema permits"
        )
        state = dfa.step(state, token_id)
    return state


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _dfa(constrain):
    """The §3 schema compiled over the fake vocabulary."""
    return constrain.compile_schema(SHOPKEEPER_SCHEMA, VOCAB)


CORPUS_PATH = (
    pathlib.Path(__file__).resolve().parents[3] / "Claude" / "Docs" / "spike" / "shopkeeper_corpus_v1.jsonl"
)


def _corpus_golds() -> list[dict[str, Any]]:
    if not CORPUS_PATH.is_file():
        return []
    golds = []
    with CORPUS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                golds.append(json.loads(line))
    return golds


CORPUS_GOLDS = _corpus_golds()


# ==========================================================================
# C1 — Schema conformance. The dim-10 feature oracle.
#
# Schema doc §5: hard-constraint violations must be 0 BY CONSTRUCTION. A
# nonzero count invalidates the spike run rather than lowering its score. These
# cells are what make that real: the decoder's output is parsed against §3 by
# this suite's independent validator, never compared to itself.
# ==========================================================================


def test_a_hostile_model_still_yields_schema_valid_output():
    """C1a — the feature oracle. A model that wants to emit junk cannot.

    The mask, not the model, is what makes the output conform. Driving the
    decoder with logits that maximally prefer a token appearing in no valid
    document is the sharpest available form of the §5 claim.
    """
    constrain = _constrain()
    dfa = _dfa(constrain)

    def hostile(_tokens):
        return logits(**{str(JUNK_ID): INT32_MAX})

    tokens = constrain.greedy_decode(dfa, hostile, max_tokens=4096)
    text = detokenize(tokens)

    assert JUNK_ID not in tokens, "the decoder emitted a token no schema state permits"
    doc = json.loads(text)  # a schema-invalid emission fails here first
    assert_schema_valid(doc)


def test_decode_reproduces_the_document_the_model_prefers():
    """C1b — the mask constrains without distorting a valid preference.

    C1a alone is satisfied by a decoder that ignores the logits entirely and
    walks a fixed path. This cell fails such a decoder: given logits that prefer
    a specific §3-valid document, the decode must produce exactly that document.
    """
    constrain = _constrain()
    dfa = _dfa(constrain)
    target_doc = make_doc(
        intent="price_query",
        artist="vale",
        day="sat",
        style="geometric",
        budget_band="under_200",
        polarity="negate",
        correction=True,
    )
    target = tokenize(canonical(target_doc))

    def preferring(tokens):
        step = len(tokens)
        if step >= len(target):
            return logits()
        return logits(**{str(target[step]): INT32_MAX})

    produced = constrain.greedy_decode(dfa, preferring, max_tokens=4096)

    assert produced == target, (
        "the decoder did not follow the model's preference through valid tokens: "
        f"{detokenize(produced)!r} != {canonical(target_doc)!r}"
    )
    assert_schema_valid(json.loads(detokenize(produced)))


def test_decode_stops_at_an_accepting_state():
    """C1c — the decode terminates on the schema, not on max_tokens.

    A decoder that runs to its token cap and happens to have emitted a valid
    document en route would pass C1a. This cell fails it: the returned sequence
    must itself be the complete document.
    """
    constrain = _constrain()
    dfa = _dfa(constrain)

    def hostile(_tokens):
        return logits(**{str(JUNK_ID): INT32_MAX})

    tokens = constrain.greedy_decode(dfa, hostile, max_tokens=4096)
    assert dfa.is_accepting(drive(dfa, tokens)), "the returned token sequence does not reach an accepting state"
    assert not dfa.is_accepting(drive(dfa, tokens[:-1])), "the decode ran past the first accepting state"


# ==========================================================================
# C2 — G-7: DFA completeness. Routed to Curie by §17.
#
# Over-constraint ships silently: a DFA that rejects a legitimate value makes
# some gold labels unreachable, and the spike reports a quality loss that is
# actually a compiler bug. The schema is small and fixed, so the sweep is
# exhaustive over every enum value in every slot.
#
# Completeness alone is satisfiable by a DFA that accepts everything — which
# would be a total constraint-engine failure passing a completeness suite. The
# soundness cells below are therefore part of this cell group, not a separate
# concern.
# ==========================================================================


@pytest.mark.parametrize(
    "document",
    SWEEP_DOCUMENTS,
    ids=[canonical(doc)[:60] for doc in SWEEP_DOCUMENTS],
)
def test_dfa_accepts_every_enum_value_in_every_field(document):
    """C2a — G-7. Every §3 enum value in every field is reachable under the mask."""
    constrain = _constrain()
    dfa = _dfa(constrain)
    assert_schema_valid(document)  # the fixture is a §3-valid document by our own oracle
    final = drive(dfa, tokenize(canonical(document)))
    assert dfa.is_accepting(final), f"the DFA does not accept the schema-valid document {canonical(document)!r}"


@pytest.mark.skipif(not CORPUS_GOLDS, reason=f"frozen corpus not present at {CORPUS_PATH}")
def test_dfa_accepts_every_gold_label_in_the_frozen_corpus():
    """C2b — G-7 against the labels the spike actually scores.

    Every gold document in the frozen corpus must be reachable under the mask.
    An unreachable gold is a compiler bug that would be reported as a slot-F1
    loss — the exact silent failure G-7 exists to catch. Exhaustive over the
    corpus, not a sample.
    """
    constrain = _constrain()
    dfa = _dfa(constrain)
    unreachable = []
    for row in CORPUS_GOLDS:
        gold = row["gold"]
        assert_schema_valid(gold)
        final = drive(dfa, tokenize(canonical(gold)))
        if not dfa.is_accepting(final):
            unreachable.append(row["id"])
    assert not unreachable, f"gold labels unreachable under the compiled DFA: {unreachable}"


# Each near-miss is DERIVED from a canonical §3-valid document by substituting exactly one value,
# rather than hardcoded. Hardcoding is how this cell silently rots: the pre-F-5 fixtures were
# four-field literals, so once §3 gained `negated_slot` the DFA rejected them for the MISSING FIELD
# rather than for the near-miss value — the cell stayed green while testing nothing it claimed.
# Deriving from make_doc() makes "differs in exactly one value" true by construction.
NEAR_MISS_CASES = [
    ("intent-not-an-enum-value", '"intent":"book"', '"intent":"booking"'),
    ("unspecified-is-not-an-intent", '"intent":"book"', '"intent":"unspecified"'),
    ("slot-value-not-in-its-enum", '"artist":"rin"', '"artist":"kim"'),
    ("polarity-not-an-enum-value", '"polarity":"affirm"', '"polarity":"maybe"'),
    # `deposit_ok` was dropped from the negated_slot enum: its own enum already carries polarity,
    # so it was unreachable. It is now a FORBIDDEN value, which makes it a soundness fixture — this
    # case is what holds the drop in place.
    ("negated_slot-not-an-enum-value", '"negated_slot":"none"', '"negated_slot":"deposit_ok"'),
    ("correction-not-a-bool", '"correction":false', '"correction":"false"'),
]


def _near_miss(find: str, replace: str) -> str:
    valid = canonical(make_doc(artist="rin", day="mon"))
    assert find in valid, f"fixture precondition: {find!r} is not in the canonical document {valid!r}"
    return valid.replace(find, replace, 1)


@pytest.mark.parametrize(
    "find,replace",
    [(find, replace) for _, find, replace in NEAR_MISS_CASES],
    ids=[name for name, _, _ in NEAR_MISS_CASES],
)
def test_dfa_rejects_a_string_the_schema_forbids(find, replace):
    """C2c — soundness. Without this, an accept-everything DFA passes C2a and C2b.

    Each fixture is a near-miss of a valid document: it differs from a §3-valid
    string in exactly one value, so rejection cannot come from gross malformation.
    """
    text = _near_miss(find, replace)
    constrain = _constrain()
    dfa = _dfa(constrain)
    state = dfa.start
    for token_id in tokenize(text):
        if token_id not in dfa.valid_tokens(state):
            return  # rejected at the first token that leaves the schema's language
        state = dfa.step(state, token_id)
    assert not dfa.is_accepting(state), f"the DFA accepts a schema-invalid string: {text!r}"


def test_dfa_does_not_accept_a_truncated_document():
    """C2d — soundness. Accepting is the end of the document, not any point in it."""
    constrain = _constrain()
    dfa = _dfa(constrain)
    tokens = tokenize(canonical(make_doc()))
    for length in range(len(tokens)):
        assert not dfa.is_accepting(drive(dfa, tokens[:length])), (
            f"the DFA accepts a {length}-token prefix of a document that is {len(tokens)} tokens long"
        )


def test_the_mask_is_non_vacuous_at_every_reachable_state():
    """C2e — soundness. A mask that permits the whole vocabulary constrains nothing."""
    constrain = _constrain()
    dfa = _dfa(constrain)
    vacuous = [state for state in reachable_states(dfa) if len(dfa.valid_tokens(state)) >= VOCAB_SIZE]
    assert not vacuous, f"states whose valid set is the entire vocabulary: {sorted(vacuous)}"


def test_valid_mask_is_the_indicator_of_valid_tokens():
    """C2f — the two mask surfaces agree.

    §9's runtime reads the bitmask; this suite's other cells read the token set.
    If they can diverge, half the suite tests a surface the runtime never uses.
    """
    constrain = _constrain()
    dfa = _dfa(constrain)
    for state in reachable_states(dfa):
        mask = dfa.valid_mask(state)
        assert len(mask) == VOCAB_SIZE, f"state {state}: mask length {len(mask)} != vocab size {VOCAB_SIZE}"
        expected = dfa.valid_tokens(state)
        actual = {i for i, allowed in enumerate(mask) if allowed}
        assert actual == set(expected), f"state {state}: valid_mask and valid_tokens disagree"


# ==========================================================================
# C3 — G-7a: the non-empty-valid-set obligation (§9, Loki 2026-07-14).
#
# If a reachable state's valid-token set were empty, the mask would zero every
# logit and argmax would return the lowest token index — the §6.5 pinned
# tie-break cannot distinguish "all masked" from "all tied". The runtime
# structurally cannot catch this, so the obligation is the compiler's.
#
# Both halves are required: the failure cell (a), and the negative control (b)
# that PROVES the runtime cannot catch it. Without (b), (a) is an unproven claim.
# ==========================================================================


def test_the_fixed_shape_schema_compiles_and_no_reachable_state_has_an_empty_valid_set():
    """C3a — the positive obligation, and the schema doc §3 "by construction" claim.

    §3 is fixed-shape (no optional keys, no recursion) precisely so every state's
    valid set is non-empty by construction. §17 dim 7 requires every "by
    construction" to map to a test or be rewritten. This is that test.

    Accepting states are terminal here — the document is complete and nothing may
    follow — so the obligation is asserted over non-accepting states. §9 states it
    over "every reachable state", which the terminal accepting state cannot satisfy
    without an EOS token in its valid set. That imprecision is a finding routed to
    the planner (see the record); this suite pins the reading it asserts against.
    """
    constrain = _constrain()
    dfa = _dfa(constrain)
    starved = [
        state
        for state in reachable_states(dfa)
        if not dfa.is_accepting(state) and not dfa.valid_tokens(state)
    ]
    assert not starved, (
        f"non-accepting states with an empty valid-token set: {sorted(starved)} — "
        "argmax would emit the lowest token index at each of them, silently"
    )


def test_compile_rejects_a_schema_state_unsatisfiable_in_this_vocabulary():
    """C3b — G-7a failure cell. Reject-over-degrade, with a which-state/why diagnostic.

    A schema satisfiable in the abstract whose required continuation cannot be
    spelled in THIS vocabulary. The compiler holds both the schema and the
    vocabulary, so the rejection is its obligation (§9).
    """
    constrain = _constrain()
    schema = {
        "type": "object",
        "properties": {"k": {"enum": ["zzz"]}},
        "required": ["k"],
        "additionalProperties": False,
    }
    vocab = tuple(sorted(set('{}":,k')))  # every character of `{"k":"zzz"}` except `z`
    assert not any("z" in piece for piece in vocab), "the fixture vocabulary must not be able to spell the value"

    with pytest.raises(constrain.SchemaCompileError) as excinfo:
        constrain.compile_schema(schema, vocab)

    error = excinfo.value
    assert error.state_id is not None, "the rejection does not identify which state is unsatisfiable"
    assert '{"k":"'.startswith(error.prefix) or error.prefix.startswith("{"), (
        f"the diagnostic's prefix {error.prefix!r} does not locate the state within the schema's serialization"
    )
    assert "zzz" in error.reason, f"the diagnostic does not say why: {error.reason!r}"


def test_the_same_schema_compiles_when_the_vocabulary_can_spell_it():
    """C3c — the paired control for C3b.

    Without this, C3b passes against a compiler that rejects every schema. The
    rejection must be caused by the vocabulary, not by the schema's shape.
    """
    constrain = _constrain()
    schema = {
        "type": "object",
        "properties": {"k": {"enum": ["zzz"]}},
        "required": ["k"],
        "additionalProperties": False,
    }
    vocab = tuple(sorted(set('{}":,kz')))

    compiled = constrain.compile_schema(schema, vocab)
    state = compiled.start
    for char in '{"k":"zzz"}':
        token_id = vocab.index(char)
        assert token_id in compiled.valid_tokens(state), f"state {state} rejects {char!r} of the only valid document"
        state = compiled.step(state, token_id)
    assert compiled.is_accepting(state)


def test_an_empty_valid_set_emits_the_lowest_index_token():
    """C3d — G-7a NEGATIVE CONTROL. The runtime cannot catch a starved state.

    A hand-built DFA (not the compiler's output) with a reachable, non-accepting
    state that permits nothing. If it reached the runtime, the mask zeroes every
    logit and the §6.5 tie-break returns the lowest token index — a
    schema-invalid token, emitted silently, with no error raised, in the one
    subsystem whose entire purpose is conformance.

    This control is what proves the compiler's C3b obligation is load-bearing
    rather than defensive. If a future implementation makes this cell fail by
    adding an emptiness check inside argmax, that is a §9 contract change (the
    obligation was placed on the compiler precisely because the runtime step is
    one bitmask AND plus an argmax — O(1), zero allocation, no branch), and it
    routes to the planner rather than being absorbed here.
    """
    constrain = _constrain()
    starved = constrain.TokenDFA(
        vocab=VOCAB,
        transitions={0: {}},
        start=0,
        accepting=frozenset(),
    )

    assert starved.valid_tokens(0) == frozenset(), "fixture error: the control state must permit nothing"
    assert not starved.is_accepting(0), "fixture error: the control state must not be accepting"

    mask = starved.valid_mask(0)
    assert not any(mask), "fixture error: the control state's mask must be all-zero"

    chosen = constrain.argmax_masked(logits(**{str(VOCAB_SIZE - 1): INT32_MAX}), mask)

    assert chosen == 0, f"argmax over an all-masked int32 logit vector returned {chosen}, expected the lowest index 0"
    assert chosen not in starved.valid_tokens(0), (
        "the control did not demonstrate the failure: the emitted token must be one the state forbids"
    )


def test_the_runtime_cannot_distinguish_an_empty_mask_from_a_legitimate_one():
    """C3e — G-7a negative control, the discriminability half.

    §9's claim is that an all-zero mask is indistinguishable from a legitimate
    one at the AND. C3d shows the starved state emits token 0; this cell shows
    that emission is byte-identical to the output of a perfectly legitimate state
    whose only valid token IS token 0 — across logit vectors that share nothing.
    There is no observable at the runtime's step boundary that separates them, so
    no runtime check is possible and the compiler's proof is the only defense.
    """
    constrain = _constrain()
    empty_mask = [False] * VOCAB_SIZE
    legitimate_mask = [i == 0 for i in range(VOCAB_SIZE)]

    probes = [
        logits(),
        logits(**{str(VOCAB_SIZE - 1): INT32_MAX}),
        logits(**{str(i): INT32_MIN for i in range(VOCAB_SIZE)}),
        logits(**{str(i): (i * 7919) % 1013 for i in range(VOCAB_SIZE)}),
    ]

    under_empty = [constrain.argmax_masked(probe, empty_mask) for probe in probes]
    under_legitimate = [constrain.argmax_masked(probe, legitimate_mask) for probe in probes]

    assert under_empty == under_legitimate == [0] * len(probes), (
        "the starved state and a legitimate single-token state produce different observables — "
        f"empty={under_empty}, legitimate={under_legitimate}"
    )


# ==========================================================================
# C4 — Greedy argmax and the pinned tie-break (§6.5, §9).
#
# int32 logits, argmax, tie-break = lowest token index, no softmax. Masks apply
# to the logits BEFORE argmax.
# ==========================================================================


def test_an_exact_tie_selects_the_lowest_token_index():
    """C4a — §6.5's pinned tie-break, on an exact tie."""
    constrain = _constrain()
    all_valid = [True] * VOCAB_SIZE
    assert constrain.argmax_masked([5] * VOCAB_SIZE, all_valid) == 0


def test_a_tie_among_valid_tokens_selects_the_lowest_valid_index():
    """C4b — the tie-break ranges over the valid set, not the vocabulary.

    Fails an implementation that argmaxes first and then checks the mask, and one
    that reports the lowest index of the whole vocabulary regardless of the mask.
    """
    constrain = _constrain()
    mask = [i in (3, 7, 11) for i in range(VOCAB_SIZE)]
    vector = [9] * VOCAB_SIZE
    assert constrain.argmax_masked(vector, mask) == 3


def test_the_mask_applies_before_argmax():
    """C4c — §9. The globally-largest logit is masked out; a smaller valid token wins."""
    constrain = _constrain()
    mask = [i in (4, 9) for i in range(VOCAB_SIZE)]
    vector = logits(**{"0": INT32_MAX, "4": 10, "9": 20})
    assert constrain.argmax_masked(vector, mask) == 9


def test_a_valid_token_at_int32_min_beats_every_masked_token():
    """C4d — masking is not "set the logit to INT32_MIN".

    A real int32 logit CAN be INT32_MIN (§17 dim 6 names INT32_MIN as a required
    numerical edge). An implementation that masks by writing a sentinel into the
    logit vector ties the masked tokens against a legitimate INT32_MIN logit, and
    the lowest-index tie-break then hands the step to a masked token. This cell is
    the one that fails such an implementation.
    """
    constrain = _constrain()
    mask = [i == VOCAB_SIZE - 1 for i in range(VOCAB_SIZE)]
    vector = [INT32_MIN] * VOCAB_SIZE
    assert constrain.argmax_masked(vector, mask) == VOCAB_SIZE - 1


def test_int32_extremes_do_not_saturate_the_selection():
    """C4e — the full int32 range is carried through the selection."""
    constrain = _constrain()
    mask = [True] * VOCAB_SIZE
    vector = [INT32_MIN] * VOCAB_SIZE
    vector[5] = INT32_MAX
    vector[6] = INT32_MAX - 1
    assert constrain.argmax_masked(vector, mask) == 5


# ==========================================================================
# C5 — Tokenizer-boundary correctness (§9 item 2).
#
# §9 compiles the DFA over TOKEN sequences, not characters, so jumps land on
# token boundaries by construction. The fixtures assert their own preconditions
# so a vocabulary change cannot make these cells vacuous.
# ==========================================================================


def test_a_multi_token_enum_value_drives_correctly():
    """C5a — a value whose tokenization splits across tokens is still driven correctly."""
    constrain = _constrain()
    dfa = _dfa(constrain)
    document = make_doc(style="geometric")
    tokens = tokenize(canonical(document))

    geo, metric = VOCAB.index("geo"), VOCAB.index("metric")
    assert geo in tokens and metric in tokens, (
        "fixture precondition: `geometric` must tokenize across multiple tokens for this cell to test anything"
    )
    assert "geometric" not in VOCAB, "fixture precondition: `geometric` must not be a single token"

    final = drive(dfa, tokens)
    assert dfa.is_accepting(final)
    assert detokenize(tokens) == canonical(document)


def test_a_token_spanning_a_value_boundary_is_accepted():
    """C5b — the retokenization hazard.

    The canonical tokenization of a §3 document contains tokens that cross a
    value's closing quote into the following punctuation and the next key
    (`","`, `":"`). A DFA whose states are pinned to value boundaries rejects
    them. §9 answers this by compiling over token sequences; this cell is the
    assertion that the answer holds.
    """
    constrain = _constrain()
    dfa = _dfa(constrain)
    document = make_doc(intent="book", artist="rin")
    tokens = tokenize(canonical(document))

    spanning = [VOCAB.index('","'), VOCAB.index('":"')]
    assert any(t in tokens for t in spanning), (
        "fixture precondition: the canonical tokenization must contain a boundary-spanning token"
    )

    final = drive(dfa, tokens)
    assert dfa.is_accepting(final)


# ==========================================================================
# C6 — Forced single-token states.
#
# The precondition jump-forward (§9 item 2) rests on. Jump-forward itself is a
# runtime mechanism the spike harness does not implement — routed to the S5
# runtime build slot; see the record for the scope call. This cell tests the
# property that is in scope here.
# ==========================================================================


def test_a_singleton_valid_set_forces_its_token_regardless_of_the_logits():
    """C6a — where the schema permits exactly one token, the model has no vote."""
    constrain = _constrain()
    dfa = _dfa(constrain)
    singletons = [
        state
        for state in reachable_states(dfa)
        if not dfa.is_accepting(state) and len(dfa.valid_tokens(state)) == 1
    ]
    assert singletons, "fixture precondition: the fixed-shape §3 schema must produce forced-token states"

    for state in singletons:
        (forced,) = tuple(dfa.valid_tokens(state))
        hostile = logits(**{str(i): INT32_MAX for i in range(VOCAB_SIZE) if i != forced})
        assert constrain.argmax_masked(hostile, dfa.valid_mask(state)) == forced, (
            f"state {state}: the forced token {forced} ({VOCAB[forced]!r}) lost to a masked token"
        )


# ==========================================================================
# C7 — A real tokenizer's vocabulary.
#
# §9's obligation is vocabulary-dependent: a schema satisfiable in the abstract
# may have a state untokenizable in THIS artifact's actual vocabulary. Every
# cell above uses the fake vocabulary; this one closes the loop on a real one.
# Skips cleanly when transformers or the cached tokenizer is unavailable.
# ==========================================================================

REAL_TOKENIZER_ID = "Qwen/Qwen2.5-1.5B-Instruct"  # §16 spike nominee


def _real_vocab() -> tuple[str, ...]:
    transformers = pytest.importorskip("transformers", reason="transformers is not installed")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(REAL_TOKENIZER_ID, local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - any load failure is a clean skip, not a red cell
        pytest.skip(f"{REAL_TOKENIZER_ID} is not available locally: {exc}")
    vocab = tokenizer.get_vocab()
    pieces = [""] * len(vocab)
    for piece, token_id in vocab.items():
        pieces[token_id] = piece
    return tuple(pieces)


def test_the_schema_compiles_over_a_real_tokenizer_vocabulary():
    """C7a — §3 is satisfiable in a real artifact's vocabulary, and no state is starved."""
    constrain = _constrain()
    vocab = _real_vocab()
    compiled = constrain.compile_schema(SHOPKEEPER_SCHEMA, vocab)

    starved = [
        state
        for state in reachable_states(compiled)
        if not compiled.is_accepting(state) and not compiled.valid_tokens(state)
    ]
    assert not starved, f"non-accepting states with an empty valid set over the real vocabulary: {sorted(starved)}"

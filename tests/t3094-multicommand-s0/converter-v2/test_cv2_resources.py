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

import signal
import time
import tracemalloc

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
    return str(info.value)


@pytest.mark.parametrize("name,schema,section", (("IntentDoc v2", F.INTENTDOC, "intents"), ("G1", F.G1, "intent_gate")),
                         ids=("intentdoc_v2", "g1"))
def test_must_admit_at_release_limits_on_pinned_vocabulary(name, schema, section, real_vocab):
    """§4.1 (3): compiles under the release limits; minItems at each array, depth, DFA states, transitions
    and section bytes equal count model v2 exactly; NFA states and tracked bytes are the converter's own
    construction and allocation accounting, recorded and held under their limits (see the record).
    Fold 4: the section-byte count includes the name blob, so the compile is given the section name."""
    vocab, pieces = real_vocab
    P.healthy_bytes(F.INTENTDOC)
    mp = P.compile_named(schema, vocab.pieces, vocab.special, section)
    counts = _counts(mp)
    model = cm.count(schema, section, pieces, F.REAL_VOCAB_SIZE)
    expected = F.model_counts_to_contract(model)
    mins = counts["minItems"]
    assert isinstance(mins, dict) and sorted(mins.values()) == sorted(model["minItems"].values()), mins
    for key in EXACT[1:]:
        assert counts[key] == expected[key], f"{name} {key}: converter {counts[key]} vs count model v2 {expected[key]}"
    for key in F.LIMIT_KEYS[1:]:
        assert counts[key] <= F.RELEASE[key], f"{name} {key} {counts[key]} over release {F.RELEASE[key]}"
    assert P.table_size(mp) == (model["dfa_states"], model["token_transitions"])


def test_must_admit_counts_at_byte_vocabulary():
    """Fast reconciliation of the same counts at the byte vocabulary (one cell for the builder's loop)."""
    P.healthy_bytes(F.INTENTDOC)
    mp = P.compile_named(F.INTENTDOC, F.BYTE_VOCAB, frozenset(), "intents")
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
    message = _refused_by(key, lambda: P.compile_schema_to_mask_pages(p["over_schema"], vocab.pieces,
                                                                       special_ids=vocab.special))
    if key == "minItems":  # fold 7: the RFC 6901 schema pointer of the over array
        pointer = "/properties/a/items/properties/b" if name == "R_MIN_INNER" else "/properties/a"
        assert F.names_pointer(message, (pointer,)), f"{name}: refusal does not name {pointer}: {message}"


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



# ------------------------------------------------------------------ fold 1: the under side of every boundary
#
# A table silently truncated at the boundary (or a minimum unrolled without its loop) can still accept the one
# example document and report the model's numbers. So on every under schema the table the artifact carries is
# reconciled with count model v2 exactly (states, transitions, accepting states), and the counts the converter
# reports must describe that table, not a model of it (§4.1 l.424 last sentence; §6 row 4 truncation mutant).

def _reconcile(mp, model, label, section=True):
    states, transitions = P.table_size(mp)
    assert (states, transitions) == (model["dfa_states"], model["token_transitions"]), \
        f"{label}: table carries {states} states / {transitions} transitions, count model v2 {model['dfa_states']} / " \
        f"{model['token_transitions']} (silently truncated or unrolled table)"
    assert len(mp.accepting) == model["accepting"], f"{label}: accepting states"
    counts = _counts(mp)
    assert (counts["dfa_states"], counts["token_transitions"]) == (states, transitions), \
        f"{label}: reported counts {counts['dfa_states']} / {counts['token_transitions']} do not describe the table"
    assert counts["recursion_depth"] == model["recursion_depth"], label
    assert sorted(counts["minItems"].values()) == sorted(model["minItems"].values()), label
    if section:
        assert counts["serialized_mask_bytes"] == model["section_bytes"], f"{label}: section bytes"


BYTE_UNDER = {
    **{f"injection_{n}": (lambda n=n: I.pair(n)) for n in I.PAIRS},
    **{f"with_min_{r}_{i}": (lambda r=r, i=i: (F.with_min(r, i), None, None, None))
       for r, i in ((1, 0), (2, 0), (3, 0), (5, 0), (1, 1), (1, 2), (1, 8))},
    "R_MIN_ROOT": lambda: (F.pair("R_MIN_ROOT")["under_schema"], None, None, None),
    "R_MIN_INNER": lambda: (F.pair("R_MIN_INNER")["under_schema"], None, None, None),
    "triple_array": lambda: (F.triple_array(), None, None, None),
    "depth_8_value": lambda: (F.deepen_value(2), None, None, None),
}


@pytest.mark.parametrize("label", tuple(BYTE_UNDER))
def test_under_side_table_reconciled_at_byte_vocabulary(label):
    """Every injection pair's under schema (at its injected limit) and every byte-vocabulary minItems and
    depth variant: the carried table equals count model v2 exactly and the reported counts describe it."""
    P.healthy_bytes(F.INTENTDOC)
    under, _, key, limit = BYTE_UNDER[label]()
    mp = P.compile_named(under, F.BYTE_VOCAB, frozenset(), "intents", {key: limit} if key else None)
    model = cm.count(under, "intents")
    _reconcile(mp, model, label)


@pytest.mark.parametrize("name", RELEASE_PAIRS)
def test_under_side_table_reconciled_on_pinned_vocabulary(name, real_vocab):
    """Every release pair's under schema at the release limits on the pinned vocabulary, reconciled with the
    frozen count model v2 counts (resource-pairs-v2.json, section name `pair`)."""
    vocab, _ = real_vocab
    P.healthy_bytes(F.INTENTDOC)
    p = F.pair(name)
    mp = P.compile_named(p["under_schema"], vocab.pieces, vocab.special, "pair")
    _reconcile(mp, p["under"], name)
    raw = B.serialize(p["under_schema"], _example(p["under_schema"]))
    ids = _content_tokens(raw, vocab)
    B.check_tokens(p["under_schema"], ids, vocab)  # Oracle B admits this tokenization (precondition)
    assert P.accepts_ids(mp, ids), f"{name}: string content spelled with a multi-byte token refused"


def _content_tokens(raw, vocab):
    """One byte per token, except each `go` string value's content is the single pinned-vocabulary token
    `go` (a table that keeps only one-byte content tokens refuses it)."""
    out, rest = [], raw
    while b'"go"' in rest:
        head, _, rest = rest.partition(b'"go"')
        out += vocab.bytewise(head + b'"') + [vocab.id_of(b"go")] + vocab.bytewise(b'"')
    return out + vocab.bytewise(rest)


# ------------------------------------------------------------------ fold 2: refusal before the crossing allocation

class _OverTime(Exception):
    pass


def _traced(fn, seconds=None):
    """(outcome, traced peak bytes, elapsed seconds) of one call; with `seconds`, the call is interrupted
    there (a converter still building past its crossing never returns in time)."""
    def stop(*_):
        raise _OverTime()
    old = signal.signal(signal.SIGALRM, stop) if seconds else None
    tracemalloc.start()
    began = time.perf_counter()
    if seconds:
        signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        outcome = fn()
    except P.SchemaCompileError as error:
        outcome = error
    except _OverTime:
        outcome = _OverTime()
    finally:
        if seconds:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)
        elapsed = time.perf_counter() - began
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return outcome, peak, elapsed


TRIVIAL = F.obj({"k": {"type": "string", "enum": ["x"]}})
PEAK_KEYS = ("tracked_temporary_bytes", "serialized_mask_bytes")


@pytest.mark.parametrize("key", PEAK_KEYS)
def test_refusal_peak_shrinks_with_the_injected_limit(key, real_vocab):
    """§4.1 l.450 and l.446: refuse "before allocating the crossing buffer" / "before section emission".
    IntentDoc v2 on the pinned vocabulary, refused with the target limit injected at L = 2/3 of its count model
    v2 count and at L/100: a converter that reserves before it allocates stops at the crossing, so the refused
    call's traced peak above the trivial schema's compile peak (the vocabulary trie and other fixed cost) is,
    at L/100, under half of what it is at L. One that builds the whole table (or emits the section) and
    compares afterwards has the same peak at both. The pinned vocabulary is where the table dominates the
    schema's fixed cost (NFA, byte DFA): at the byte vocabulary the NFA alone is most of the peak, so no
    converter could show the difference there for section bytes."""
    vocab, pieces = real_vocab
    P.healthy_real(F.INTENTDOC, vocab)
    base, base_peak, _ = _traced(lambda: P.compile_schema_to_mask_pages(TRIVIAL, vocab.pieces,
                                                                       special_ids=vocab.special))
    assert not isinstance(base, Exception), f"trivial control refused: {base}"
    model = F.model_counts_to_contract(cm.count(F.INTENTDOC, "intents", pieces, F.REAL_VOCAB_SIZE))
    limit = model[key] * 2 // 3
    peaks = {}
    for divisor in (1, 100):
        outcome, peak, _ = _traced(lambda: P.compile_schema_to_mask_pages(
            F.INTENTDOC, vocab.pieces, special_ids=vocab.special, limits={key: limit // divisor}))
        assert isinstance(outcome, P.SchemaCompileError) and F.DIAGNOSTICS[key] in str(outcome), \
            f"{key} at L/{divisor}: not refused by its own guard ({outcome!r})"
        peaks[divisor] = peak - base_peak
    assert peaks[100] < peaks[1] / 2, \
        f"{key}: refused-call peak above the trivial compile {peaks[100]:,} B at L/100 vs {peaks[1]:,} B at L " \
        f"(built before the guard spoke)"


NFA_FAR_OVER_FACTOR = 100
FAR_OVER_PEAK_MARGIN = 32 * 2 ** 20  # bytes above the trivial compile's peak
FAR_OVER_TIME_MARGIN = 10.0  # seconds above the trivial compile's time


def far_over_nfa_schema():
    """IntentDoc v2 with 4,100 extra 102-byte `action` values: count model v2's NFA is about 105 times the
    release NFA limit (checked by test_cv2_reference)."""
    return I._extend_enum("action", 4100, 100)


def test_far_over_nfa_schema_refused_before_determinization(real_vocab):
    """§4.1 l.450, §6 row 11: a schema 100x over its NFA limit at the pinned vocabulary is refused by the NFA
    guard inside a stated budget: traced peak at most the trivial schema's compile peak + 32 MiB, time at most
    its time + 10 s. Building the NFA through and then determinizing it (the post-hoc construction) takes
    hundreds of MiB and tens of seconds, so it is over budget even though its refusal is named correctly."""
    vocab, _ = real_vocab
    P.healthy_bytes(F.INTENTDOC)
    base, base_peak, base_time = _traced(lambda: P.compile_schema_to_mask_pages(TRIVIAL, vocab.pieces,
                                                                               special_ids=vocab.special))
    assert not isinstance(base, Exception), f"trivial control refused: {base}"
    budget = base_time + FAR_OVER_TIME_MARGIN
    outcome, peak, elapsed = _traced(lambda: P.compile_schema_to_mask_pages(
        far_over_nfa_schema(), vocab.pieces, special_ids=vocab.special), seconds=budget + 1)
    assert not isinstance(outcome, _OverTime), f"still compiling after {budget:.1f} s: no refusal at the crossing"
    assert isinstance(outcome, P.SchemaCompileError) and F.DIAGNOSTICS["nfa_states"] in str(outcome), \
        f"not refused by the NFA guard: {outcome!r}"
    assert peak <= base_peak + FAR_OVER_PEAK_MARGIN, \
        f"refused-call peak {peak:,} B over budget {base_peak + FAR_OVER_PEAK_MARGIN:,} B (built past the crossing)"
    assert elapsed <= budget, f"refused after {elapsed:.1f} s, budget {budget:.1f} s"


# ------------------------------------------------------------------ N1: injected limits do not persist

def test_injected_limit_does_not_persist():
    """An injected limit binds only its own call: inject minItems 1 (refusing an inner minimum of 2), then the
    same schema compiles at the release limits."""
    P.healthy_bytes(F.INTENTDOC)
    schema = F.with_min(1, 2)
    _refused_by("minItems", lambda: P.compile_bytes_limited(schema, {"minItems": 1}))
    mp = P.compile_bytes(schema)
    assert P.accepts_raw_bytes(mp, B.serialize(schema, F.doc(*F.intents(1, 2))))

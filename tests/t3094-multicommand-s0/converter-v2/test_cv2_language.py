"""Item 1 — accepted language of IntentDoc v2 (plan §4.1 converter paragraph; §6 rows 4, 7, 8, 11; §7 Oracle B).

Product cells compile the frozen schema through `compile_schema_to_mask_pages` and walk the compiled token
table. Expected verdicts come from Oracle B (cv2_oracle), never from converter output."""
from __future__ import annotations

import random
import re

import pytest

import cv2_cases as C
import cv2_fixtures as F
import cv2_oracle as B
import cv2_product as P
import cv2_tokens as T
import contract as A


# ------------------------------------------------------------------ byte vocabulary

@pytest.mark.parametrize("case,raw", C.ACCEPT, ids=[c[0] for c in C.ACCEPT])
def test_byte_vocab_accepts_canonical_document(case, raw):
    mp = P.healthy_bytes(F.INTENTDOC)
    assert P.accepts_raw_bytes(mp, raw), f"{case}: canonical IntentDoc v2 document refused"


@pytest.mark.parametrize("case,raw", C.REJECT, ids=[c[0] for c in C.REJECT])
def test_byte_vocab_refuses_noncanonical_document(case, raw):
    mp = P.healthy_bytes(F.INTENTDOC)
    assert not P.accepts_raw_bytes(mp, raw), f"{case}: non-canonical document accepted"


@pytest.mark.parametrize("base,raw", C.MUTATION_BASES, ids=[c[0] for c in C.MUTATION_BASES])
def test_byte_vocab_single_byte_mutation_sweep_matches_oracle_b(base, raw):
    """Every single-byte deletion, insertion (from a structural alphabet) and substitution of a canonical
    document: the converter's verdict equals Oracle B's. Catches over- and under-acceptance at every offset,
    including both array loops, the empty-array exit and the item/atom separators."""
    mp = P.healthy_bytes(F.INTENTDOC)
    mutants = set()
    for i in range(len(raw) + 1):
        if i < len(raw):
            mutants.add(raw[:i] + raw[i + 1:])
        for b in C.MUTATION_ALPHABET:
            mutants.add(raw[:i] + bytes([b]) + raw[i:])
            if i < len(raw) and raw[i] != b:
                mutants.add(raw[:i] + bytes([b]) + raw[i + 1:])
    disagreements = [m for m in sorted(mutants) if P.accepts_raw_bytes(mp, m) != B.accepts(F.INTENTDOC, m)]
    assert not disagreements, f"{len(disagreements)} of {len(mutants)} mutants disagree, e.g. {disagreements[:3]!r}"


def _random_complete_walks(mp, count, seed, max_tokens=4000):
    """Random walks over the compiled table that steer toward acceptance once long enough."""
    rows = mp.transitions
    distance = {s: 0 for s in mp.accepting}
    changed = True
    while changed:
        changed = False
        for s, row in rows.items():
            best = min((distance[t] + 1 for t in row.values() if t in distance), default=None)
            if best is not None and distance.get(s, 1 << 30) > best:
                distance[s] = best
                changed = True
    rng = random.Random(seed)
    outputs = []
    for _ in range(count):
        state, out = mp.start, []
        while state not in mp.accepting or (rows.get(state) and rng.random() < 0.2):
            row = rows.get(state, {})
            assert row, f"walk reached a non-accepting state {state} with an empty row"
            if len(out) > max_tokens // 4 or rng.random() < 0.15:
                tid = min(row, key=lambda t: (distance.get(row[t], 1 << 30), t))
            else:
                tid = rng.choice(sorted(row))
            out.append(tid)
            state = row[tid]
            assert len(out) < max_tokens, "walk did not terminate"
        outputs.append(out)
    return outputs


def test_byte_vocab_random_compiled_outputs_pass_oracles_b_and_a():
    mp = P.healthy_bytes(F.INTENTDOC)
    outputs = _random_complete_walks(mp, 300, seed=3094)
    lengths = []
    for ids in outputs:
        raw = bytes(ids)
        value, _ = B.parse(F.INTENTDOC, raw)
        A.shape(A.parse(raw), "v2")
        lengths.append((len(value["intents"]), max((len(i["conditions"]) for i in value["intents"]), default=0)))
    assert any(n >= 2 for n, _ in lengths) and any(m >= 2 for _, m in lengths), \
        "random walks never repeated either loop; the sample does not exercise both arrays"


def test_byte_vocab_every_reachable_state_can_finish_and_accepting_ends_document():
    """§4.1 l.422 and §6 row 5 (dead end), S0a fold 3a: every reachable state reaches an accepting state, not
    merely has a non-empty row (a cycle with no exit has non-empty rows); the canonical language is
    prefix-free, so an accepting state admits no further token (whole-document termination, §6 row 7)."""
    mp = P.healthy_bytes(F.INTENTDOC)
    dead = P.cannot_finish(mp)
    assert not dead, f"{len(dead)} reachable states cannot reach an accepting state"
    assert all(not mp.transitions.get(s) for s in mp.accepting & P.reachable(mp))


# ------------------------------------------------------------------ fold 3b: unsatisfiable tables are refused
#
# Conductor ruling (brief-s0-fold1, 3b): §4.1's "nonaccepting-empty-row refusal" is read at the class level:
# the converter refuses, by name and with no artifact, any schema whose token DFA has a reachable state from
# which no accepting state is reachable; the empty row is its zero-step case. Contract assumed for the name: the
# message says the state cannot reach acceptance, matched by DEAD_END (and names no resource guard).

DEAD_END = re.compile(r"accept|dead[ -]?end|unsatisfiable|empty valid-token", re.IGNORECASE)
NO_CLOSE = tuple(b"" if i == ord("]") else p for i, p in enumerate(F.BYTE_VOCAB))
UNSPELLABLE_EXIT = {"close_bracket_zeroed": NO_CLOSE, "close_bracket_only_in_close_brace": NO_CLOSE + (b"]}",)}


@pytest.mark.parametrize("label", tuple(UNSPELLABLE_EXIT))
def test_unspellable_array_exit_refused_by_name(label):
    """IntentDoc v2 over a vocabulary that cannot spell the conditions array's exit: the table would loop on
    `,` with no accepting state reachable (checked by test_cv2_reference), so the compile is refused."""
    P.healthy_bytes(F.INTENTDOC)
    with pytest.raises(P.SchemaCompileError) as info:
        P.compile_schema_to_mask_pages(F.INTENTDOC, UNSPELLABLE_EXIT[label], special_ids=frozenset())
    message = str(info.value)
    assert "unsupported construct" not in message, f"{P.RED_ARRAY}: {message}"
    assert DEAD_END.search(message), f"refusal does not say the table cannot reach acceptance: {message}"
    assert not any(d in message for d in F.DIAGNOSTICS.values()), message


@pytest.mark.parametrize("schema_name", ("intentdoc_v2", "g1"))
@pytest.mark.parametrize("level", ("byte", "pinned"))
def test_admitted_tables_can_always_finish(schema_name, level, request):
    """The other side of fold 3b: IntentDoc v2 and G1 are admitted at both vocabularies, and every reachable
    state of the admitted table reaches an accepting state."""
    P.healthy_bytes(F.INTENTDOC)
    schema = F.INTENTDOC if schema_name == "intentdoc_v2" else F.G1
    if level == "byte":
        mp = P.healthy_bytes(schema)
    else:
        mp = P.healthy_real(schema, request.getfixturevalue("real_vocab")[0])
    dead = P.cannot_finish(mp)
    assert not dead, f"{schema_name} at {level}: {len(dead)} reachable states cannot reach acceptance"


@pytest.mark.parametrize("case", T.CASES, ids=[c[0] for c in T.CASES])
def test_byte_vocab_with_compound_pieces_token_boundaries(case):
    """Byte vocabulary plus the boundary compounds: admitted compounds land where the one-byte walk lands;
    opening compounds are absent from the state's row while the one-byte quote is present."""
    name, raw, offset, piece, verdict = case
    vocab = F.BYTE_VOCAB + T.COMPOUNDS
    mp = _compiled_with(vocab)
    tid = vocab.index(piece)
    ok, state, _ = P.walk_ids(mp, list(raw[:offset]))
    assert ok, f"{name}: prefix before the boundary refused"
    row = mp.transitions[state]
    if verdict:
        assert tid in row, f"{name}: compound {piece!r} is masked at its boundary"
        after_compound = row[tid]
        ok, after_bytes, _ = P.walk_ids(mp, list(raw[:offset + len(piece)]))
        assert ok and after_compound == after_bytes, f"{name}: compound destination differs from the byte walk"
        ids = list(raw[:offset]) + [tid] + list(raw[offset + len(piece):])
        assert P.accepts_ids(mp, ids)
    else:
        assert tid not in row, f"{name}: opening compound {piece!r} admitted"
        assert piece[0] in row, f"{name}: ordinary one-byte control masked"


_WITH = {}


def _compiled_with(vocab):
    key = vocab
    if key not in _WITH:
        try:
            _WITH[key] = P.compile_schema_to_mask_pages(F.INTENTDOC, vocab, special_ids=frozenset())
        except P.SchemaCompileError as error:
            pytest.fail(f"{P.RED_ARRAY}: {error}")
    return _WITH[key]


# ------------------------------------------------------------------ pinned real vocabulary

REAL_ACCEPT = [c for c in C.ACCEPT if c[0] in ("one_intent_empty_conditions", "three_intents_1_0_3_atoms_loop_in_loop",
                                               "utf8_two_three_four_byte", "unicode_escapes_both_cases")]
REAL_REJECT = [c for c in C.REJECT if c[0] in ("root_empty_array", "trailing_comma_intents", "trailing_comma_conditions",
                                               "comma_only_conditions", "whitespace_between_items", "reversed_atom_keys",
                                               "insignificant_leaf_structural", "premature_termination")]


@pytest.mark.parametrize("case,raw", REAL_ACCEPT + REAL_REJECT, ids=[c[0] for c in REAL_ACCEPT + REAL_REJECT])
def test_real_vocab_bytewise_verdict_matches_oracle_b(case, raw, real_vocab):
    vocab, _ = real_vocab
    mp = P.healthy_real(F.INTENTDOC, vocab)
    assert P.accepts_ids(mp, vocab.bytewise(raw)) == B.accepts(F.INTENTDOC, raw), case


@pytest.mark.parametrize("case", T.CASES, ids=[c[0] for c in T.CASES])
def test_real_vocab_token_boundaries_on_actual_ids(case, real_vocab):
    """§6 row 8: actual token IDs of the pinned tokenizer for both crossing directions at the intent item
    boundary and the atom boundary, with the ordinary one-byte control."""
    vocab, _ = real_vocab
    name, raw, offset, piece, verdict = case
    mp = P.healthy_real(F.INTENTDOC, vocab)
    tid = vocab.id_of(piece)
    ok, state, _ = P.walk_ids(mp, vocab.bytewise(raw[:offset]))
    assert ok, f"{name}: prefix refused"
    row = mp.transitions[state]
    if verdict:
        assert tid in row, f"{name}: token {tid} {piece!r} masked"
        ok, after_bytes, _ = P.walk_ids(mp, vocab.bytewise(raw[:offset + len(piece)]))
        assert ok and row[tid] == after_bytes, f"{name}: compound destination differs"
        assert P.accepts_ids(mp, vocab.ids(T.pieces(raw, offset, piece)))
    else:
        assert tid not in row, f"{name}: opening token {tid} {piece!r} admitted"
        assert vocab.id_of(piece[:1]) in row, f"{name}: ordinary one-byte control masked"


def test_real_vocab_no_special_id_in_any_row(real_vocab):
    vocab, _ = real_vocab
    mp = P.healthy_real(F.INTENTDOC, vocab)
    hits = {s: sorted(set(row) & vocab.special) for s, row in mp.transitions.items() if set(row) & vocab.special}
    assert not hits, f"special ids admitted at {len(hits)} states"


def spelled_special_pieces(vocab):
    """The pinned vocabulary with every added (special) token spelled as its literal content bytes, the way a
    raw tokenizer table carries it. count_model_v2.load_vocab blanks those pieces, so a cell compiling the
    `real_vocab` pieces cannot see whether the converter excludes special ids itself (S1 builder finding, M17)."""
    import json
    pieces = list(vocab.pieces)
    for added in json.loads((F.TOKENIZER_DIR / "tokenizer.json").read_text(encoding="utf-8"))["added_tokens"]:
        if added["id"] < len(pieces):
            pieces[added["id"]] = added["content"].encode("utf-8")
    return pieces


def test_real_vocab_spelled_special_ids_excluded_by_the_converter(real_vocab):
    """§4.1: special ids are excluded structurally by the converter, from whatever bytes the caller passes.
    With the specials spelled out (ordinary printable bytes, valid free-string content), no row admits one."""
    vocab, _ = real_vocab
    P.healthy_real(F.INTENTDOC, vocab)
    pieces = spelled_special_pieces(vocab)
    assert all(pieces[i] for i in vocab.special)
    try:
        mp = P.compile_schema_to_mask_pages(F.INTENTDOC, pieces, special_ids=vocab.special)
    except P.SchemaCompileError as error:
        pytest.fail(f"{P.RED_ARRAY}: {error}")
    hits = sorted({t for row in mp.transitions.values() for t in row} & vocab.special)
    assert not hits, f"special ids {hits[:5]} admitted (of {len(vocab.special)})"


def test_real_vocab_random_compiled_outputs_pass_oracle_b_token_rules(real_vocab):
    vocab, _ = real_vocab
    mp = P.healthy_real(F.INTENTDOC, vocab)
    for ids in _random_complete_walks(mp, 60, seed=8087, max_tokens=3000):
        B.check_tokens(F.INTENTDOC, ids, vocab)

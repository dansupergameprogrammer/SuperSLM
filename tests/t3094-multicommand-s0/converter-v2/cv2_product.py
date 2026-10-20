"""Thin access to the product converter for the converter-v2 cells. Walks use only the compiled artifact's
public table (`transitions`, `start`, `accepting`), never `MaskPages.accepts`."""
from __future__ import annotations

import functools
import json

import pytest

import cv2_fixtures as F
from sslm_convert_schema import SchemaCompileError, compile_schema_to_mask_pages  # noqa: F401

RED_ARRAY = "red-unimplemented: the converter does not compile an array schema yet"


def _key(schema):
    return json.dumps(schema)  # insertion order is the serialization order: never sort_keys


def compile_bytes(schema, **kw):
    return compile_schema_to_mask_pages(schema, F.BYTE_VOCAB, special_ids=frozenset(), **kw)


RED_LIMITS = "red-unimplemented: the converter takes no limits argument"


def compile_bytes_limited(schema, limits):
    """Compile with injected limits; a converter without the argument is red-unimplemented."""
    try:
        return compile_bytes(schema, limits=limits)
    except TypeError as error:
        pytest.fail(f"{RED_LIMITS}: {error}")


@functools.lru_cache(maxsize=64)
def _cached_bytes(key):
    try:
        return compile_bytes(json.loads(key)), None
    except SchemaCompileError as error:
        return None, error


def healthy_bytes(schema):
    """A healthy compile at the byte vocabulary, or the cell is red-unimplemented with the refusal named."""
    mp, error = _cached_bytes(_key(schema))
    if mp is None:
        pytest.fail(f"{RED_ARRAY}: {error}")
    return mp


_REAL = {}


def healthy_real(schema, vocab):
    key = _key(schema)
    if key not in _REAL:
        try:
            _REAL[key] = (compile_schema_to_mask_pages(schema, vocab.pieces, special_ids=vocab.special), None)
        except SchemaCompileError as error:
            _REAL[key] = (None, error)
    mp, error = _REAL[key]
    if mp is None:
        pytest.fail(f"{RED_ARRAY} (pinned vocabulary): {error}")
    return mp


def walk_ids(mp, token_ids):
    """(admitted every token, final state or the state where the walk stopped, index of the refused token)."""
    state = mp.start
    for index, tid in enumerate(token_ids):
        row = mp.transitions.get(state, {})
        if tid not in row:
            return False, state, index
        state = row[tid]
    return True, state, None


def accepts_ids(mp, token_ids):
    ok, state, _ = walk_ids(mp, token_ids)
    return ok and state in mp.accepting


def accepts_raw_bytes(mp, raw):
    return accepts_ids(mp, list(raw))


RED_NAME = "red-unimplemented: the converter takes no section name argument"


def compile_named(schema, pieces, special, name, limits=None):
    """Compile with the section name the section-byte count depends on (fold 4: the model counts the name
    blob, so a compile-time section-byte count needs the name). `limits` is passed only when given."""
    kw = {"special_ids": special, "name": name}
    if limits:
        kw["limits"] = limits
    try:
        return compile_schema_to_mask_pages(schema, pieces, **kw)
    except TypeError as error:
        pytest.fail(f"{RED_NAME}: {error}")
    except SchemaCompileError as error:
        if "unsupported construct" in str(error):
            pytest.fail(f"{RED_ARRAY}: {error}")
        raise AssertionError(f"a schema the cell must admit was refused: {error}") from error


def table_size(mp):
    """(states, transitions) of the table the artifact actually carries."""
    return len(mp.transitions), sum(map(len, mp.transitions.values()))


def reachable(mp):
    seen, frontier = {mp.start}, [mp.start]
    while frontier:
        for t in mp.transitions.get(frontier.pop(), {}).values():
            if t not in seen:
                seen.add(t)
                frontier.append(t)
    return seen


def cannot_finish(mp):
    """Reachable states from which no accepting state is reachable (fold 3a)."""
    reach = reachable(mp)
    back = {}
    for s in reach:
        for t in mp.transitions.get(s, {}).values():
            back.setdefault(t, set()).add(s)
    live, frontier = set(mp.accepting & reach), list(mp.accepting & reach)
    while frontier:
        for s in back.get(frontier.pop(), ()):
            if s not in live:
                live.add(s)
                frontier.append(s)
    return reach - live

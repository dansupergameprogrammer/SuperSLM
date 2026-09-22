# T-2919 (TE-372 S3) PROVENANCE: vendored verbatim from the private records worktree
# `Claude/Vitruvius/t2912-probe/sslm_convert_schema_value_level.py` (Wizard repo,
# superslm-super-embedder-fixes-c20ddf branch) into this repo so the compiler test suite is
# self-contained and reproducible from a fresh clone -- no line below this comment block was
# edited from the source file. See `../PROVENANCE.md` for the full vendoring manifest, and
# specifically for why a comparison against this module is no longer, by itself, a claim of
# correctness now that `tools/sslm_convert_schema.py` implements the same T-2912 design T-2915
# ported: this module remains a useful cross-implementation consistency check and a MUTANT
# source (T-2914's rejected Arm A comparison, the dropped-value-history mutant), but it is not
# an independent oracle for a fresh correctness claim. This module loads `../t2911-probe/
# sslm_convert_schema_close_structural.py` as its own base at import time (`_BASE_PATH`,
# relative to `__file__`) -- keep both directories siblings under `reference/` or that load
# breaks.
"""T-2912 probe compiler: close the punctuation/whitespace non-answer class at value level.

This module loads T-2911's complete byte-level compiler and replaces only the string-leaf
construction and token walk.  The string leaf carries a monotone two-state semantic property:

* U: the decoded value has not yet contributed a meaningful byte;
* M: at least one decoded value byte is neither JSON structural punctuation (``{}[],:``) nor
  JSON whitespace (space, tab, carriage return, line feed).

Only M has a closing-quote transition.  U classifies raw ASCII bytes, short JSON escapes, and
``\\uXXXX`` escapes by the decoded byte.  Every non-ASCII scalar is meaningful because each byte of
its UTF-8 encoding lies outside the ten-byte insignificant set.  M is absorbing.  Token boundaries
therefore cannot change the accepted-value language.

T-2910's strict value-open rule and special-token exclusion remain.  T-2911's per-token structural
close refusal is removed: once the value is in M, natural compound closes remain fully admitted.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Mapping, Sequence


_BASE_PATH = Path(__file__).resolve().parents[1] / "t2911-probe" / "sslm_convert_schema_close_structural.py"
_SPEC = importlib.util.spec_from_file_location("_t2911_value_level_base", _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load T-2911 compiler from {_BASE_PATH}")
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

SchemaCompileError = _BASE.SchemaCompileError
MaskPages = _BASE.MaskPages
zero_special_ids = _BASE.zero_special_ids

# Six JSON structural punctuation bytes plus RFC 8259 JSON whitespace.
INSIGNIFICANT_VALUE_BYTES = frozenset(b"{}[],: \t\r\n")
_INSIGNIFICANT_UESCAPE_CODEPOINTS = frozenset(INSIGNIFICANT_VALUE_BYTES)
_HEX_BYTES = tuple(ord(c) for c in "0123456789abcdefABCDEF")


def _add_string_leaf(edges, add_node, cursor: int, content_nodes: set[int]) -> int:
    """Build T-2908's UTF-8-valid string leaf with U/M value-history modes."""
    quote, backslash = 0x22, 0x5C

    def node() -> int:
        result = add_node()
        content_nodes.add(result)
        return result

    def edge(state: int, lo: int, hi: int, target: int) -> None:
        for byte_val in range(lo, hi + 1):
            edges[state].setdefault(byte_val, set()).add(target)

    # M is T-2908's original content automaton.  It is absorbing until the close.
    m_sc = node()
    m_lead2 = node()
    edge(m_sc, 0xC2, 0xDF, m_lead2)
    edge(m_lead2, 0x80, 0xBF, m_sc)

    m_lead3a, m_lead3b, m_lead3c, m_mid3 = node(), node(), node(), node()
    edge(m_sc, 0xE0, 0xE0, m_lead3a)
    edge(m_sc, 0xE1, 0xEC, m_lead3b)
    edge(m_sc, 0xED, 0xED, m_lead3c)
    edge(m_sc, 0xEE, 0xEF, m_lead3b)
    edge(m_lead3a, 0xA0, 0xBF, m_mid3)
    edge(m_lead3b, 0x80, 0xBF, m_mid3)
    edge(m_lead3c, 0x80, 0x9F, m_mid3)
    edge(m_mid3, 0x80, 0xBF, m_sc)

    m_lead4a, m_lead4b, m_lead4c, m_mid4a, m_mid4b = node(), node(), node(), node(), node()
    edge(m_sc, 0xF0, 0xF0, m_lead4a)
    edge(m_sc, 0xF1, 0xF3, m_lead4b)
    edge(m_sc, 0xF4, 0xF4, m_lead4c)
    edge(m_lead4a, 0x90, 0xBF, m_mid4a)
    edge(m_lead4b, 0x80, 0xBF, m_mid4a)
    edge(m_lead4c, 0x80, 0x8F, m_mid4a)
    edge(m_mid4a, 0x80, 0xBF, m_mid4b)
    edge(m_mid4b, 0x80, 0xBF, m_sc)

    for byte_val in range(0x20, 0x80):
        if byte_val not in (quote, backslash):
            edges[m_sc].setdefault(byte_val, set()).add(m_sc)

    m_escape = node()
    edges[m_sc].setdefault(backslash, set()).add(m_escape)
    for escape_char in _BASE._SHORT_ESCAPE_CHARS:
        edges[m_escape].setdefault(ord(escape_char), set()).add(m_sc)
    m_hex = [node() for _ in range(4)]
    edges[m_escape].setdefault(ord("u"), set()).add(m_hex[0])
    for index, state in enumerate(m_hex):
        target = m_hex[index + 1] if index + 1 < len(m_hex) else m_sc
        for digit in _HEX_BYTES:
            edges[state].setdefault(digit, set()).add(target)

    # U starts immediately after the opening quote and has no closing transition.
    u_sc = node()
    edges[cursor].setdefault(quote, set()).add(u_sc)
    for byte_val in range(0x20, 0x80):
        if byte_val in (quote, backslash):
            continue
        target = u_sc if byte_val in INSIGNIFICANT_VALUE_BYTES else m_sc
        edges[u_sc].setdefault(byte_val, set()).add(target)

    # A non-ASCII scalar is meaningful under the byte language as soon as its valid lead arrives.
    edge(u_sc, 0xC2, 0xDF, m_lead2)
    edge(u_sc, 0xE0, 0xE0, m_lead3a)
    edge(u_sc, 0xE1, 0xEC, m_lead3b)
    edge(u_sc, 0xED, 0xED, m_lead3c)
    edge(u_sc, 0xEE, 0xEF, m_lead3b)
    edge(u_sc, 0xF0, 0xF0, m_lead4a)
    edge(u_sc, 0xF1, 0xF3, m_lead4b)
    edge(u_sc, 0xF4, 0xF4, m_lead4c)

    u_escape = node()
    edges[u_sc].setdefault(backslash, set()).add(u_escape)
    decoded_short = {
        '"': 0x22, "\\": 0x5C, "/": 0x2F, "b": 0x08,
        "f": 0x0C, "n": 0x0A, "r": 0x0D, "t": 0x09,
    }
    for escape_char, decoded in decoded_short.items():
        target = u_sc if decoded in INSIGNIFICANT_VALUE_BYTES else m_sc
        edges[u_escape].setdefault(ord(escape_char), set()).add(target)

    insignificant_hex = tuple(f"{cp:04x}" for cp in sorted(_INSIGNIFICANT_UESCAPE_CODEPOINTS))
    generic_remaining: dict[int, int] = {}

    def significant_remainder(remaining: int) -> int:
        if remaining == 0:
            return m_sc
        if remaining not in generic_remaining:
            state = node()
            generic_remaining[remaining] = state
            target = significant_remainder(remaining - 1)
            for digit in _HEX_BYTES:
                edges[state].setdefault(digit, set()).add(target)
        return generic_remaining[remaining]

    prefix_nodes: dict[str, int] = {}

    def unicode_prefix(prefix: str) -> int:
        if prefix in prefix_nodes:
            return prefix_nodes[prefix]
        state = node()
        prefix_nodes[prefix] = state
        for digit in _HEX_BYTES:
            normalized = chr(digit).lower()
            candidate = prefix + normalized
            if len(candidate) == 4:
                target = u_sc if int(candidate, 16) in _INSIGNIFICANT_UESCAPE_CODEPOINTS else m_sc
            elif any(value.startswith(candidate) for value in insignificant_hex):
                target = unicode_prefix(candidate)
            else:
                target = significant_remainder(4 - len(candidate))
            edges[state].setdefault(digit, set()).add(target)
        return state

    edges[u_escape].setdefault(ord("u"), set()).add(unicode_prefix(""))

    following = add_node()  # literal skeleton; deliberately not a content node
    edges[m_sc].setdefault(quote, set()).add(following)
    return following


def _token_targets(dfa: Sequence[Mapping[int, int]], state: int, root,
                   content_states: frozenset[int]) -> dict[int, int]:
    """T-2910 open-side strictness, with no token-local close blacklist."""
    targets: dict[int, int] = {}
    stack = [(root, state, 0, False)]
    while stack:
        trie_node, current, depth, stop_after = stack.pop()
        if trie_node.token_id is not None:
            targets[trie_node.token_id] = current
        if stop_after:
            continue
        row = dfa[current]
        cur_is_content = current in content_states
        if len(row) < len(trie_node.children):
            pairs = ((byte_val, trie_node.children.get(byte_val), target)
                     for byte_val, target in row.items())
        else:
            pairs = ((byte_val, child, row.get(byte_val))
                     for byte_val, child in trie_node.children.items())
        for byte_val, child, target in pairs:
            if child is None or target is None:
                continue
            opening = (not cur_is_content) and target in content_states
            if opening and depth != 0:
                continue
            stack.append((child, target, depth + 1, opening))
    return targets


# Patch only the private extension points used by the inherited compiler pipeline.
_BASE._add_string_leaf = _add_string_leaf
_BASE._token_targets = _token_targets

compile_schema_to_mask_pages = _BASE.compile_schema_to_mask_pages
_groups = _BASE._groups
_char_dfa = _BASE._char_dfa
_vocab_trie = _BASE._vocab_trie

__all__ = [
    "SchemaCompileError", "MaskPages", "compile_schema_to_mask_pages", "zero_special_ids",
    "INSIGNIFICANT_VALUE_BYTES",
]

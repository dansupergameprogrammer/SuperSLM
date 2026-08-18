"""sslm_convert_schema.py -- G5-1: converter-side schema compiler (T-2132/T-2119, design
`Claude/Vitruvius/t2119-g5-constrained-decoding-design-2026-08-16.md` Sec4/Sec6 G5-1, Wizard repo).

Offline converter tooling (Sec4: "the schema compiler is a converter component... like every
other Sec11 converter stage"). Compiles a D-SLM45 per-field schema subset (objects with known,
required, in-order keys; enums; booleans; no `additionalProperties`) into a token-level DFA with
per-state valid-token bitmask pages, over a caller-supplied vocabulary. Proves at compile time that
every reachable non-accepting state has a non-empty valid-token set (G-7a) and rejects the schema
otherwise with a which-state/why diagnostic -- a reject-over-degrade rejection, never a degraded
runtime check (D-SLM40's positive requirement: the runtime never checks for an all-masked vector).

The DFA-construction algorithm (schema -> alternation groups -> character NFA/DFA -> token-level
DFA via a vocabulary trie) is this project's own already-proven construction, first built as the
T-066 spike reference (`Claude/Vitruvius/t2119-repair-verification/constrain.py`, itself grounded
in the shipped reference decoder `tests/reference/superslm_spike/constrain.py`) and re-executed
against the design's own strike/repair verification (design Sec10.3). This module is an
independent production port of that construction into the converter's own module surface
(`tools/sslm_convert_schema.py`, following the `sslm_convert_adapter.py`/`sslm_convert_manifest.py`
naming convention) -- it does not import the spike script (a records-tree file, not a shipped
dependency), it reproduces the same proven algorithm as first-class converter code.

Mask pages: `MaskPages.page(state)` returns a packed bitmask (bytes, one bit per vocabulary token
id, little-endian within each byte) -- the artifact-resident, per-state, read-only representation
design Sec4's architecture table names ("Table lookup + bitmask AND, int32 logits, pre-argmax,
indexed by the sequence's own DFA-state field"). `MaskPages.accepts(s)` is this module's own
conformance-fuzz convenience (greedy token-trie walk driven by the compiled DFA's own valid-token
row at each state) -- not a runtime primitive, used only by this suite's own G5-1 fuzz cells.
"""
from __future__ import annotations

import json
from collections import deque
from typing import Any, Mapping, Sequence

__all__ = [
    "SchemaCompileError",
    "MaskPages",
    "compile_schema_to_mask_pages",
]


class SchemaCompileError(Exception):
    """G-7a's rejection: a reachable non-accepting state has an empty valid-token set.

    `state_id` names the offending compiled-DFA state; `prefix` is the canonical-
    serialization prefix that reaches it; `reason` explains, in words, what continuation
    no vocabulary token can spell. `state_id` is `None` only for a structural rejection
    (a schema construct outside the D-SLM45 compilable subset) -- no DFA state exists yet
    to name in that case.
    """

    def __init__(
        self,
        message: str,
        *,
        state_id: int | None = None,
        prefix: str = "",
        reason: str = "",
    ) -> None:
        super().__init__(message)
        self.state_id = state_id
        self.prefix = prefix
        self.reason = reason


# --- schema -> ordered alternation groups (D-SLM45's compilable subset) -----------------


def _groups(schema: Mapping[str, Any], path: str) -> list[tuple[str, ...]]:
    """`schema`'s canonical serialization as an ordered list of alternation groups.

    Each group is the set of strings the serialization may take at that position; a
    literal is a group of one. Concatenating one choice from each group, in order,
    produces exactly the canonical serializations D-SLM40 admits for this schema.
    """
    if "enum" in schema:
        values = schema["enum"]
        if not values:
            raise SchemaCompileError(f"empty enum at {path}", reason=f"empty enum at {path}")
        return [tuple(json.dumps(v, ensure_ascii=False) for v in values)]

    node_type = schema.get("type")

    if node_type == "boolean":
        return [("true", "false")]

    if node_type == "object":
        properties = schema.get("properties") or {}
        required = list(schema.get("required", list(properties)))
        if schema.get("additionalProperties", False) is not False:
            raise SchemaCompileError(
                f"object at {path} permits additional properties",
                reason="additionalProperties must be false; an open object has no fixed serialization",
            )
        if required != list(properties):
            raise SchemaCompileError(
                f"object at {path} has optional or reordered keys",
                reason="every property must be required, in properties order (D-SLM45); optional/reordered keys are outside the compilable subset",
            )
        groups: list[tuple[str, ...]] = [("{",)]
        for index, (key, sub_schema) in enumerate(properties.items()):
            if index:
                groups.append((",",))
            groups.append((json.dumps(key, ensure_ascii=False) + ":",))
            groups.extend(_groups(sub_schema, f"{path}.{key}"))
        groups.append(("}",))
        return groups

    raise SchemaCompileError(
        f"unsupported construct at {path}",
        reason=f"type {node_type!r} is outside D-SLM45's compilable subset (objects with known keys, enums, booleans; cross-field constraints are scored, not compiled)",
    )


# --- the character-level DFA (thompson-construction-by-hand over the alternation groups) --


def _char_dfa(groups: Sequence[Sequence[str]]) -> tuple[list[dict[str, int]], int, frozenset[int]]:
    edges: list[dict[str, set[int]]] = [{}]

    def add_node() -> int:
        edges.append({})
        return len(edges) - 1

    cursor = 0
    for group in groups:
        following = add_node()
        for alternative in group:
            if not alternative:
                raise SchemaCompileError(
                    "an alternation contains the empty string",
                    reason="a zero-length alternative makes the serialization ambiguous",
                )
            node = cursor
            for index, char in enumerate(alternative):
                target = following if index == len(alternative) - 1 else add_node()
                edges[node].setdefault(char, set()).add(target)
                node = target
        cursor = following
    accept_node = cursor

    start_set = frozenset({0})
    ids: dict[frozenset[int], int] = {start_set: 0}
    dfa: list[dict[str, int]] = [{}]
    accepting: set[int] = set()
    queue: deque[frozenset[int]] = deque([start_set])
    while queue:
        current = queue.popleft()
        state = ids[current]
        if accept_node in current:
            accepting.add(state)
        moves: dict[str, set[int]] = {}
        for node in current:
            for char, targets in edges[node].items():
                moves.setdefault(char, set()).update(targets)
        row: dict[str, int] = {}
        for char in sorted(moves):
            target_set = frozenset(moves[char])
            if target_set not in ids:
                ids[target_set] = len(dfa)
                dfa.append({})
                queue.append(target_set)
            row[char] = ids[target_set]
        dfa[state] = row
    return dfa, 0, frozenset(accepting)


def _shortest_completion(dfa: Sequence[Mapping[str, int]], accepting: frozenset[int], state: int) -> str:
    seen = {state}
    queue: deque[tuple[int, str]] = deque([(state, "")])
    while queue:
        current, text = queue.popleft()
        if current in accepting:
            return text
        for char in sorted(dfa[current]):
            target = dfa[current][char]
            if target not in seen:
                seen.add(target)
                queue.append((target, text + char))
    return ""


# --- the vocabulary trie (drives the token-level DFA construction) ------------------------


class _TrieNode:
    __slots__ = ("children", "token_id")

    def __init__(self) -> None:
        self.children: dict[str, _TrieNode] = {}
        self.token_id: int | None = None


def _vocab_trie(vocab: Sequence[str]) -> _TrieNode:
    root = _TrieNode()
    for token_id, piece in enumerate(vocab):
        if not piece:
            continue
        node = root
        for char in piece:
            child = node.children.get(char)
            if child is None:
                child = _TrieNode()
                node.children[char] = child
            node = child
        if node.token_id is None:  # first (lowest-index) token spelling this piece wins
            node.token_id = token_id
    return root


def _token_targets(dfa: Sequence[Mapping[str, int]], state: int, root: _TrieNode) -> dict[int, int]:
    """Every vocabulary token drivable from `state`, and the character-DFA state each reaches."""
    targets: dict[int, int] = {}
    stack: list[tuple[_TrieNode, int]] = [(root, state)]
    while stack:
        node, current = stack.pop()
        if node.token_id is not None:
            targets[node.token_id] = current
        row = dfa[current]
        if len(row) < len(node.children):
            for char, target in row.items():
                child = node.children.get(char)
                if child is not None:
                    stack.append((child, target))
        else:
            for char, child in node.children.items():
                target = row.get(char)
                if target is not None:
                    stack.append((child, target))
    return targets


# --- mask pages -----------------------------------------------------------------------------


class MaskPages:
    """The compiled artifact: a token-level DFA plus per-state valid-token bitmask pages.

    `transitions[state]` maps every valid token id at `state` to the state it reaches --
    the same table the runtime's O(1) lookup-plus-bitmask-AND step consults (design Sec4).
    A state absent from `transitions`, or mapped to an empty row, is either accepting (no
    further tokens required) or -- for any reachable non-accepting state -- was already
    rejected at compile time (G-7a); no such state survives into a `MaskPages` instance.
    """

    def __init__(
        self,
        vocab: Sequence[str],
        transitions: Mapping[int, Mapping[int, int]],
        start: int,
        accepting: frozenset[int],
    ) -> None:
        self.vocab = tuple(vocab)
        self.transitions: dict[int, dict[int, int]] = {s: dict(row) for s, row in transitions.items()}
        self.start = start
        self.accepting = frozenset(accepting)

    def valid_token_ids(self, state: int) -> frozenset[int]:
        return frozenset(self.transitions.get(state, {}))

    def page(self, state: int) -> bytes:
        """The state's valid-token set, packed one bit per vocabulary token id (LSB-first
        within each byte) -- the artifact-resident bitmask page design Sec4 names."""
        n_bytes = (len(self.vocab) + 7) // 8
        out = bytearray(n_bytes)
        for token_id in self.transitions.get(state, {}):
            out[token_id >> 3] |= 1 << (token_id & 7)
        return bytes(out)

    def step(self, state: int, token_id: int) -> int:
        return self.transitions[state][token_id]

    def is_accepting(self, state: int) -> bool:
        return state in self.accepting

    def accepts(self, s: str) -> bool:
        """G5-1 fuzz-suite convenience: does `s` walk the compiled DFA, token by token
        (greedy longest-match over each state's own valid-token row), to an accepting
        state with no leftover input? Not a runtime primitive -- the runtime never
        tokenizes a raw string against the mask; it consults `page(state)` against
        logits the model already produced. This walk exists so the fuzz suite can
        assert admitted/non-admitted strings without a full decode loop."""
        state = self.start
        pos = 0
        n = len(s)
        while pos < n:
            row = self.transitions.get(state, {})
            best: tuple[str, int] | None = None
            for token_id, next_state in row.items():
                piece = self.vocab[token_id]
                if piece and s.startswith(piece, pos):
                    if best is None or len(piece) > len(best[0]):
                        best = (piece, next_state)
            if best is None:
                return False
            piece, state = best
            pos += len(piece)
        return state in self.accepting


# --- the compiler ---------------------------------------------------------------------------


def compile_schema_to_mask_pages(schema: Mapping[str, Any], vocab: Sequence[str]) -> MaskPages:
    """Compile `schema` (D-SLM45's per-field subset) to a token-level DFA with per-state
    valid-token bitmask pages, over `vocab`.

    Raises `SchemaCompileError` -- naming the offending state, the serialization prefix
    that reaches it, and the continuation no vocabulary token can spell -- for any
    reachable non-accepting state with an empty valid-token set (G-7a). This is the
    compiler's own proof obligation: an all-zero mask is indistinguishable from a
    legitimate one at the runtime's AND step (D-SLM40), so the guarantee must be
    structural, here, at compile time -- the runtime must not (and, by this design, does
    not) check for it.
    """
    vocab = tuple(vocab)
    char_dfa, char_start, char_accepting = _char_dfa(_groups(schema, "$"))
    trie = _vocab_trie(vocab)

    ids: dict[int, int] = {char_start: 0}
    prefixes: dict[int, str] = {0: ""}
    transitions: dict[int, dict[int, int]] = {}
    accepting: set[int] = set()
    queue: deque[int] = deque([char_start])
    while queue:
        current = queue.popleft()
        state = ids[current]
        if current in char_accepting:
            accepting.add(state)
        targets = _token_targets(char_dfa, current, trie)
        if not targets and current not in char_accepting:
            prefix = prefixes[state]
            completion = _shortest_completion(char_dfa, char_accepting, current)
            reason = (
                f"no token in the vocabulary can spell the required continuation {completion!r} "
                f"after the serialization prefix {prefix!r}"
            )
            raise SchemaCompileError(
                f"state {state} has an empty valid-token set: {reason}",
                state_id=state,
                prefix=prefix,
                reason=reason,
            )
        row: dict[int, int] = {}
        for token_id in sorted(targets):
            target = targets[token_id]
            if target not in ids:
                ids[target] = len(ids)
                prefixes[ids[target]] = prefixes[state] + vocab[token_id]
                queue.append(target)
            row[token_id] = ids[target]
        transitions[state] = row

    return MaskPages(vocab, transitions, 0, frozenset(accepting))

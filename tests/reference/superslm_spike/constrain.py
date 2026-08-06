"""Schema-constrained decoding for the T-066 spike harness.

Realizes SuperSLM_Plan.md §9 mechanism 1 (schema FSM masks) and §6.5 (greedy
argmax over int32 logits, tie-break = lowest token index) in offline Python.

The compiled language is the canonical serialization of the schema: schema key
order, no insignificant whitespace, ``true``/``false`` for booleans.

The DFA is compiled over token sequences rather than characters, so a token that
spans a value boundary drives the same states a character walk would reach.
"""

from __future__ import annotations

import json
from collections import deque
from typing import Any, Callable, Mapping, Sequence

__all__ = [
    "SchemaCompileError",
    "TokenDFA",
    "compile_schema",
    "argmax_masked",
    "greedy_decode",
]


class SchemaCompileError(Exception):
    """A schema the compiler rejects rather than compiling to a degraded DFA.

    ``state_id`` is ``None`` when the rejection is structural — the schema uses a
    construct outside §9's compilable subset, so no state exists to name.
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


class TokenDFA:
    """A token-level DFA over a fixed vocabulary.

    ``transitions`` maps a state to the token ids it permits and the state each
    one reaches. A state absent from ``transitions``, or mapped to an empty row,
    permits nothing.
    """

    def __init__(
        self,
        vocab: Sequence[str],
        transitions: Mapping[int, Mapping[int, int]],
        start: int,
        accepting: frozenset[int],
    ) -> None:
        self.vocab = tuple(vocab)
        self.transitions = {state: dict(row) for state, row in transitions.items()}
        self.start = start
        self.accepting = frozenset(accepting)

    def valid_tokens(self, state: int) -> frozenset[int]:
        return frozenset(self.transitions.get(state, {}))

    def valid_mask(self, state: int) -> list[bool]:
        allowed = self.transitions.get(state, {})
        return [token_id in allowed for token_id in range(len(self.vocab))]

    def step(self, state: int, token_id: int) -> int:
        return self.transitions[state][token_id]

    def is_accepting(self, state: int) -> bool:
        return state in self.accepting


# --- schema -> the sequence of alternation groups it serializes to ------------


def _groups(schema: Mapping[str, Any], path: str) -> list[tuple[str, ...]]:
    """The canonical serialization of ``schema`` as an ordered list of alternations.

    Each group is the set of strings the serialization may take at that position;
    a literal is a group of one. Concatenating one choice from each group in order
    produces exactly the canonical serializations of the schema's documents.
    """
    if "enum" in schema:
        values = schema["enum"]
        if not values:
            raise SchemaCompileError(f"empty enum at {path}", reason=f"empty enum at {path}")
        return [tuple(json.dumps(value, ensure_ascii=False) for value in values)]

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
                reason="every property must be required, in properties order; optional keys are outside the compilable subset",
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
        reason=f"type {node_type!r} is outside §9's compilable subset (objects with known keys, enums, booleans)",
    )


# --- the character-level DFA --------------------------------------------------


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
    queue = deque([start_set])
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
    queue = deque([(state, "")])
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


# --- the vocabulary trie ------------------------------------------------------


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
        if node.token_id is None:
            node.token_id = token_id
    return root


def _token_targets(dfa: Sequence[Mapping[str, int]], state: int, root: _TrieNode) -> dict[int, int]:
    """Every token drivable from ``state``, and the state each one reaches."""
    targets: dict[int, int] = {}
    stack = [(root, state)]
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


# --- the compiler -------------------------------------------------------------


def compile_schema(schema: Mapping[str, Any], vocab: Sequence[str]) -> TokenDFA:
    """Compile ``schema`` to a token-level DFA over ``vocab``.

    Raises ``SchemaCompileError`` naming the state and the continuation it cannot
    spell when a reachable non-accepting state has no valid token. The proof is
    the compiler's because the runtime cannot make it: an all-zero mask is
    indistinguishable from a legitimate one at the AND (§9).
    """
    vocab = tuple(vocab)
    char_dfa, char_start, char_accepting = _char_dfa(_groups(schema, "$"))
    trie = _vocab_trie(vocab)

    ids: dict[int, int] = {char_start: 0}
    prefixes: dict[int, str] = {0: ""}
    transitions: dict[int, dict[int, int]] = {}
    accepting: set[int] = set()
    queue = deque([char_start])
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

    return TokenDFA(vocab, transitions, 0, frozenset(accepting))


# --- the runtime --------------------------------------------------------------


def argmax_masked(logits: Sequence[int], mask: Sequence[bool]) -> int:
    """The §6.5 greedy step: argmax over the masked int32 logits, lowest index wins ties.

    Masked tokens are excluded from the scan rather than assigned a sentinel
    logit. INT32_MIN is a legitimate logit value (§17 dim 6), so a sentinel would
    tie a real logit against the masked tokens and hand the step to one of them.

    An all-masked vector returns index 0 and raises nothing. This is the §9
    contract: the step is one lookup and one AND, and an all-zero mask is
    indistinguishable from a legitimate single-token mask here — which is why the
    non-empty-valid-set proof is `compile_schema`'s obligation, not this
    function's check.
    """
    best_id = 0
    best_logit: int | None = None
    for token_id, allowed in enumerate(mask):
        if not allowed:
            continue
        logit = logits[token_id]
        if best_logit is None or logit > best_logit:
            best_id = token_id
            best_logit = logit
    return best_id


def greedy_decode(
    dfa: TokenDFA,
    logits_fn: Callable[[list[int]], Sequence[int]],
    *,
    max_tokens: int,
) -> list[int]:
    """Decode greedily under ``dfa``'s masks, stopping at the first accepting state."""
    tokens: list[int] = []
    state = dfa.start
    while len(tokens) < max_tokens and not dfa.is_accepting(state):
        token_id = argmax_masked(logits_fn(list(tokens)), dfa.valid_mask(state))
        tokens.append(token_id)
        state = dfa.step(state, token_id)
    return tokens

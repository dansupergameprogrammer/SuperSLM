# T-2919 (TE-372 S3) PROVENANCE: vendored verbatim from the private records worktree
# `Claude/Vitruvius/t2908-probe/sslm_convert_schema_bytelevel.py` (Wizard repo,
# superslm-super-embedder-fixes-c20ddf branch) into this repo so the compiler test suite is
# self-contained and reproducible from a fresh clone -- no line below this comment block was
# edited from the source file. See `../PROVENANCE.md` for the full vendoring manifest. This
# module is a fixed historical design artifact (T-2908's own byte-level redesign, later
# superseded by T-2910/T-2911/T-2912 below); it is not maintained going forward and is loaded
# only as a reference/mutant for `t2908_byte_level_red.py` and `t2910_boundary_red.py`.
"""T-2908 probe: byte-level redesign of tools/sslm_convert_schema.py's string leaf (S2) plus
a full-keyword-surface refusal (S3). Prototype only, to compile and probe against the real
vocabulary; the diff this file specifies against the shipped module is what a builder applies.

S2 fix. The shipped module's alphabet is Python `str` (Unicode codepoints); the vocabulary
producer decodes each token's raw bytes to `str` with errors="replace" first. A byte-level BPE
token can be a fragment of a multi-byte UTF-8 sequence, so that decode is lossy: many distinct
byte-fragment token ids collapse to the same replacement-character string, and the trie's
lowest-id-wins rule then makes all but one of them unreachable, while the one that IS reachable
is admitted as if it were one ordinary Unicode character -- when the bytes it actually
contributes at runtime may not be valid UTF-8 at all.

The fix moves the alphabet from Unicode codepoints to raw bytes, at every layer that currently
iterates `str` characters:
  - the vocabulary is `Sequence[bytes]` (each token's raw bytes, undecoded) instead of
    `Sequence[str]`;
  - `_vocab_trie` walks each token's raw bytes;
  - `_char_dfa`'s literal-chain builder encodes each literal alternative to UTF-8 bytes before
    walking it (this is exact and lossless for literals, which are always compiler-known ASCII
    or non-ASCII text produced by `json.dumps`, never model output);
  - the string leaf's content state `S_c` is no longer a single self-loop over ~1.1M codepoints.
    It becomes an 11-state byte-level UTF-8-validity sub-automaton (the standard construction:
    a lead-byte state per length class, with the first continuation byte's admitted range
    narrowed exactly where needed to exclude overlong encodings, surrogates and codepoints past
    U+10FFFF, and shared "one more ordinary continuation byte" tail states) composed with the
    existing ASCII escape/`"`/`\\` handling, which is unaffected because every JSON structural
    byte (`"`, `\\`, C0 controls) is single-byte ASCII (0x00-0x7F) and is therefore never
    confused with a UTF-8 lead or continuation byte (0x80-0xFF).

`_token_targets`, `compile_schema_to_mask_pages` and `MaskPages` change NOT AT ALL: they are
already generic over whatever alphabet the character-level DFA and trie use (dict keys), and
byte values are as valid a dict key as single-character strings. Only `_groups` (keyword
allowlist, S3), `_char_dfa`'s literal loop, `_add_string_leaf`, `_vocab_trie`, and
`_shortest_completion` (its diagnostic text is now built over bytes) change.

S3 fix. `_groups()` gains a per-branch keyword allowlist, derived directly from the keys each
branch actually reads (its own accepted-keyword surface): enum -> {"enum"}; boolean -> {"type"};
string -> {"type"}; object -> {"type", "properties", "required", "additionalProperties"}. Any
other key present on that schema node -- `maxLength` (kept as its own, more specific reason),
`minLength`, `pattern`, `format`, `const`, or any keyword not yet invented -- is refused, naming
the keyword. This is the module's own stated law ("a reject-over-degrade rejection, never a
degraded runtime check") applied to schema ingestion instead of DFA construction: an allowlist
closes the class for every keyword this compiler has not implemented, not only the ones named so
far, including a future JSON-Schema keyword nobody has thought to blacklist yet.
"""
from __future__ import annotations

import json
import time
from collections import deque
from typing import Any, Callable, Mapping, Sequence

__all__ = ["SchemaCompileError", "MaskPages", "compile_schema_to_mask_pages"]


class SchemaCompileError(Exception):
    def __init__(self, message: str, *, state_id: int | None = None, prefix: bytes = b"", reason: str = "") -> None:
        super().__init__(message)
        self.state_id = state_id
        self.prefix = prefix
        self.reason = reason


class _StringLeaf:
    __slots__ = ()


_STRING_LEAF = _StringLeaf()
_SHORT_ESCAPE_CHARS = ('"', "\\", "/", "b", "f", "n", "r", "t")
_HEX_DIGITS = "0123456789abcdefABCDEF"
_Group = tuple[Any, ...]

# --- S3: the compiler's own accepted-keyword surface, by branch -----------------------------

_ALLOWED_KEYWORDS: dict[str, frozenset[str]] = {
    "enum": frozenset({"enum"}),
    "boolean": frozenset({"type"}),
    "string": frozenset({"type"}),
    "object": frozenset({"type", "properties", "required", "additionalProperties"}),
}

_KEYWORD_REASON_OVERRIDES: dict[str, str] = {
    "maxLength": (
        "D-SLM7435 rules only that no length bound is added to the DFA -- an unbounded "
        "self-loop ships regardless of what value a schema names -- and says nothing about "
        "schema-ingestion; silently accepting the keyword and compiling an unbounded field "
        "anyway would discard the schema author's own stated constraint with no diagnostic, "
        "wrong independent of whether the bound is ever implemented (T-2859 F2, D-SLM7462)"
    ),
}


def _reject_unimplemented_keywords(schema: Mapping[str, Any], path: str, allowed: frozenset[str]) -> None:
    for key in sorted(schema):
        if key in allowed:
            continue
        reason = _KEYWORD_REASON_OVERRIDES.get(key, (
            f"{key!r} is a JSON-Schema keyword this compiler does not implement; a schema "
            f"naming it is rejected rather than silently accepted with the constraint "
            f"unenforced (generalizes the maxLength precedent, T-2859 F2/S3, to every "
            f"keyword outside {sorted(allowed)!r})"
        ))
        raise SchemaCompileError(f"{key} is not supported at {path}", reason=reason)


def _groups(schema: Mapping[str, Any], path: str) -> list[_Group]:
    if "enum" in schema:
        _reject_unimplemented_keywords(schema, path, _ALLOWED_KEYWORDS["enum"])
        values = schema["enum"]
        if not values:
            raise SchemaCompileError(f"empty enum at {path}", reason=f"empty enum at {path}")
        return [tuple(json.dumps(v, ensure_ascii=False) for v in values)]

    node_type = schema.get("type")

    if node_type == "boolean":
        _reject_unimplemented_keywords(schema, path, _ALLOWED_KEYWORDS["boolean"])
        return [("true", "false")]

    if node_type == "string":
        _reject_unimplemented_keywords(schema, path, _ALLOWED_KEYWORDS["string"])
        return [(_STRING_LEAF,)]

    if node_type == "object":
        _reject_unimplemented_keywords(schema, path, _ALLOWED_KEYWORDS["object"])
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
        groups: list[_Group] = [("{",)]
        for index, (key, sub_schema) in enumerate(properties.items()):
            if index:
                groups.append((",",))
            groups.append((json.dumps(key, ensure_ascii=False) + ":",))
            groups.extend(_groups(sub_schema, f"{path}.{key}"))
        groups.append(("}",))
        return groups

    raise SchemaCompileError(
        f"unsupported construct at {path}",
        reason=f"type {node_type!r} is outside D-SLM45's compilable subset (objects with known keys, enums, booleans, unbounded free-text strings; cross-field constraints are scored, not compiled)",
    )


# --- S2: the byte-level content sub-automaton ------------------------------------------------


def _add_string_leaf(edges: list[dict[int, set[int]]], add_node: Callable[[], int], cursor: int) -> int:
    """T-2853's cyclic sub-automaton, over raw bytes. `S_c` is now a UTF-8-validity state
    machine rather than a single self-loop: every admitted path through it, from `S_c` back
    to `S_c`, consumes exactly one well-formed UTF-8 encoded codepoint's bytes -- never a lone
    continuation byte, an overlong encoding, a surrogate, or a codepoint past U+10FFFF -- so
    every string this automaton admits is valid UTF-8 by construction, and every valid UTF-8
    byte sequence that is not `"`, `\\`, or a C0 control byte is admitted from `S_c`."""
    s_c = add_node()
    edges[cursor].setdefault(0x22, set()).add(s_c)  # opening quote

    quote, backslash = 0x22, 0x5C

    def edge(state: int, lo: int, hi: int, target: int) -> None:
        for b in range(lo, hi + 1):
            edges[state].setdefault(b, set()).add(target)

    # Ordinary single-byte ASCII content: not '"', not '\', not a C0 control (0x00-0x1F).
    edge(s_c, 0x20, quote - 1, s_c)
    edge(s_c, quote + 1, backslash - 1, s_c)
    edge(s_c, backslash + 1, 0x7F, s_c)

    # Multi-byte UTF-8 lead bytes, each routed to a state whose admitted continuation range
    # excludes the overlong/surrogate/out-of-range spellings the plain 0x80-0xBF range would
    # otherwise let through.
    lead2 = add_node()
    edge(s_c, 0xC2, 0xDF, lead2)  # 0xC0-0xC1 (overlong) and bare continuation bytes: no edge
    edge(lead2, 0x80, 0xBF, s_c)

    lead3_a, lead3_b, lead3_c, mid3 = add_node(), add_node(), add_node(), add_node()
    edge(s_c, 0xE0, 0xE0, lead3_a)
    edge(s_c, 0xE1, 0xEC, lead3_b)
    edge(s_c, 0xED, 0xED, lead3_c)
    edge(s_c, 0xEE, 0xEF, lead3_b)
    edge(lead3_a, 0xA0, 0xBF, mid3)  # excludes the overlong 0x80-0x9F continuation
    edge(lead3_b, 0x80, 0xBF, mid3)
    edge(lead3_c, 0x80, 0x9F, mid3)  # excludes the surrogate range 0xA0-0xBF (U+D800-DFFF)
    edge(mid3, 0x80, 0xBF, s_c)

    lead4_a, lead4_b, lead4_c, mid4a, mid4b = add_node(), add_node(), add_node(), add_node(), add_node()
    edge(s_c, 0xF0, 0xF0, lead4_a)
    edge(s_c, 0xF1, 0xF3, lead4_b)
    edge(s_c, 0xF4, 0xF4, lead4_c)
    edge(lead4_a, 0x90, 0xBF, mid4a)  # excludes the overlong 0x80-0x8F continuation
    edge(lead4_b, 0x80, 0xBF, mid4a)
    edge(lead4_c, 0x80, 0x8F, mid4a)  # excludes codepoints past U+10FFFF
    edge(mid4a, 0x80, 0xBF, mid4b)
    edge(mid4b, 0x80, 0xBF, s_c)
    # 0xF5-0xFF: no edge (no valid codepoint starts with these bytes).

    s_e = add_node()
    edges[s_c].setdefault(backslash, set()).add(s_e)
    for escape_char in _SHORT_ESCAPE_CHARS:
        edges[s_e].setdefault(ord(escape_char), set()).add(s_c)

    hex_states = [add_node() for _ in range(4)]
    edges[s_e].setdefault(ord("u"), set()).add(hex_states[0])
    for index, state in enumerate(hex_states):
        target = hex_states[index + 1] if index + 1 < len(hex_states) else s_c
        for digit in _HEX_DIGITS:
            edges[state].setdefault(ord(digit), set()).add(target)

    following = add_node()
    edges[s_c].setdefault(quote, set()).add(following)
    return following


def _char_dfa(groups: Sequence[_Group]) -> tuple[list[dict[int, int]], int, frozenset[int]]:
    edges: list[dict[int, set[int]]] = [{}]

    def add_node() -> int:
        edges.append({})
        return len(edges) - 1

    cursor = 0
    for group in groups:
        if len(group) == 1 and group[0] is _STRING_LEAF:
            cursor = _add_string_leaf(edges, add_node, cursor)
            continue
        following = add_node()
        for alternative in group:
            if not alternative:
                raise SchemaCompileError(
                    "an alternation contains the empty string",
                    reason="a zero-length alternative makes the serialization ambiguous",
                )
            raw = alternative.encode("utf-8")
            node = cursor
            for index, byte_val in enumerate(raw):
                target = following if index == len(raw) - 1 else add_node()
                edges[node].setdefault(byte_val, set()).add(target)
                node = target
        cursor = following
    accept_node = cursor

    start_set = frozenset({0})
    ids: dict[frozenset[int], int] = {start_set: 0}
    dfa: list[dict[int, int]] = [{}]
    accepting: set[int] = set()
    queue: deque[frozenset[int]] = deque([start_set])
    while queue:
        current = queue.popleft()
        state = ids[current]
        if accept_node in current:
            accepting.add(state)
        moves: dict[int, set[int]] = {}
        for node in current:
            for byte_val, targets in edges[node].items():
                moves.setdefault(byte_val, set()).update(targets)
        row: dict[int, int] = {}
        for byte_val in sorted(moves):
            target_set = frozenset(moves[byte_val])
            if target_set not in ids:
                ids[target_set] = len(dfa)
                dfa.append({})
                queue.append(target_set)
            row[byte_val] = ids[target_set]
        dfa[state] = row
    return dfa, 0, frozenset(accepting)


def _shortest_completion(dfa: Sequence[Mapping[int, int]], accepting: frozenset[int], state: int) -> bytes:
    seen = {state}
    queue: deque[tuple[int, bytes]] = deque([(state, b"")])
    while queue:
        current, data = queue.popleft()
        if current in accepting:
            return data
        for byte_val in sorted(dfa[current]):
            target = dfa[current][byte_val]
            if target not in seen:
                seen.add(target)
                queue.append((target, data + bytes([byte_val])))
    return b""


class _TrieNode:
    __slots__ = ("children", "token_id")

    def __init__(self) -> None:
        self.children: dict[int, _TrieNode] = {}
        self.token_id: int | None = None


def _vocab_trie(vocab: Sequence[bytes]) -> _TrieNode:
    root = _TrieNode()
    for token_id, piece in enumerate(vocab):
        if not piece:
            continue
        node = root
        for byte_val in piece:
            child = node.children.get(byte_val)
            if child is None:
                child = _TrieNode()
                node.children[byte_val] = child
            node = child
        if node.token_id is None:
            node.token_id = token_id
    return root


def _token_targets(dfa: Sequence[Mapping[int, int]], state: int, root: _TrieNode) -> dict[int, int]:
    targets: dict[int, int] = {}
    stack: list[tuple[_TrieNode, int]] = [(root, state)]
    while stack:
        node, current = stack.pop()
        if node.token_id is not None:
            targets[node.token_id] = current
        row = dfa[current]
        if len(row) < len(node.children):
            for byte_val, target in row.items():
                child = node.children.get(byte_val)
                if child is not None:
                    stack.append((child, target))
        else:
            for byte_val, child in node.children.items():
                target = row.get(byte_val)
                if target is not None:
                    stack.append((child, target))
    return targets


class MaskPages:
    def __init__(self, vocab: Sequence[bytes], transitions: Mapping[int, Mapping[int, int]], start: int, accepting: frozenset[int]) -> None:
        self.vocab = tuple(vocab)
        self.transitions: dict[int, dict[int, int]] = {s: dict(row) for s, row in transitions.items()}
        self.start = start
        self.accepting = frozenset(accepting)

    def valid_token_ids(self, state: int) -> frozenset[int]:
        return frozenset(self.transitions.get(state, {}))

    def page(self, state: int) -> bytes:
        n_bytes = (len(self.vocab) + 7) // 8
        out = bytearray(n_bytes)
        for token_id in self.transitions.get(state, {}):
            out[token_id >> 3] |= 1 << (token_id & 7)
        return bytes(out)

    def step(self, state: int, token_id: int) -> int:
        return self.transitions[state][token_id]

    def is_accepting(self, state: int) -> bool:
        return state in self.accepting

    def accepts(self, b: bytes) -> bool:
        state = self.start
        pos = 0
        n = len(b)
        while pos < n:
            row = self.transitions.get(state, {})
            best: tuple[bytes, int] | None = None
            for token_id, next_state in row.items():
                piece = self.vocab[token_id]
                if piece and b.startswith(piece, pos):
                    if best is None or len(piece) > len(best[0]):
                        best = (piece, next_state)
            if best is None:
                return False
            piece, state = best
            pos += len(piece)
        return state in self.accepting


def compile_schema_to_mask_pages(schema: Mapping[str, Any], vocab: Sequence[bytes]) -> MaskPages:
    vocab = tuple(vocab)
    char_dfa, char_start, char_accepting = _char_dfa(_groups(schema, "$"))
    trie = _vocab_trie(vocab)

    ids: dict[int, int] = {char_start: 0}
    prefixes: dict[int, bytes] = {0: b""}
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
            raise SchemaCompileError(f"state {state} has an empty valid-token set: {reason}", state_id=state, prefix=prefix, reason=reason)
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

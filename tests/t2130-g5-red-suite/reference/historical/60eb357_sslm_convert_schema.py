# Frozen historical fixture: source commit 60eb357; source path tools/sslm_convert_schema.py.
# Original source SHA-256: d6a14c9c70e2356ef0395a5a437f1f6499f3d8d2cff255ac0d57712776a48b5b.
# This fixture is test input; do not edit. t2921_frozen_fixture_integrity.py pins it.
"""sslm_convert_schema.py -- G5-1: converter-side schema compiler (T-2132/T-2119, design
`Claude/Vitruvius/t2119-g5-constrained-decoding-design-2026-08-16.md` Sec4/Sec6 G5-1, Wizard repo).

Offline converter tooling (Sec4: "the schema compiler is a converter component... like every
other Sec11 converter stage"). Compiles a D-SLM45 per-field schema subset (objects with known,
required, in-order keys; enums; booleans; unbounded free-text strings; no `additionalProperties`)
into a token-level DFA with per-state valid-token bitmask pages, over a caller-supplied vocabulary.
Proves at compile time that every reachable non-accepting state has a non-empty valid-token set
(G-7a) and rejects the schema otherwise with a which-state/why diagnostic -- a reject-over-degrade
rejection, never a degraded runtime check (D-SLM40's positive requirement: the runtime never checks
for an all-masked vector).

A `"type": "string"` leaf (T-2853, `Claude/Vitruvius/t2853-schema-string-fields-design-2026-09-19.md`
Sec3.2, Wizard repo, D-SLM7434/D-SLM7442) compiles to a small cyclic byte sub-automaton rather than
the forward-only literal chain every other leaf produces. This is the plan's own final Sec3.9
design, closed across four folds (`Claude/Plans/te266-gpu-path.md` Sec3.9.1):

- **The alphabet is bytes, not Unicode codepoints (T-2908, folding code review TE-365 S2).** The
  vocabulary is `Sequence[bytes]` (each token's own undecoded bytes) rather than `Sequence[str]`;
  `_vocab_trie` walks each token's raw bytes; `_char_dfa`'s literal-chain builder encodes each
  literal alternative to UTF-8 bytes before walking it. The string leaf's content is an 11-state
  byte-level UTF-8-validity sub-automaton (one lead-byte state per length class, continuation
  ranges narrowed to exclude overlong encodings, surrogates, and codepoints past U+10FFFF) composed
  with the ASCII escape/`"`/`\\` handling every JSON string needs.
- **Every keyword this compiler does not implement is refused, naming itself, not only
  `maxLength` (T-2908, folding TE-365 S3, Sec3.9.6).** A per-branch keyword allowlist, derived from
  the keys each branch actually reads, refuses any other key on that schema node before doing
  anything else with it.
- **A value opens only through a token that is the bare boundary byte and nothing else; special or
  control tokens are never admitted as content anywhere (T-2910, folding TE-366's fracture,
  D-SLM7589).** `_token_targets` refuses a token that crosses from a non-content state into a
  content state past its own first byte -- an opening crossing is admitted only at depth 0. Every
  tokenizer "added"/special id is zeroed out of the compiler's own vocabulary before the trie is
  built (`zero_special_ids`).
- **The exclusion is structural, not a step a caller can forget (T-2917, folding TE-370 C1,
  D-SLM7600).** `compile_schema_to_mask_pages` takes `special_ids` as a required keyword-only
  argument and applies `zero_special_ids` to its own copy of `vocab` itself, before the trie is
  built -- the only in-repo real-vocabulary producer (`tools/t2132_build_g5_fixture.py::
  _real_vocab`) previously returned every tokenizer special id spelled out as literal bytes, and
  nothing forced a caller compiling a string-leaf schema to zero them first, so a caller who
  forgot admitted every special id as string content (22 of 22, executed on the real
  Qwen2.5-0.5B-Instruct tokenizer). Omitting the argument now raises `TypeError` naming
  `special_ids`, at the call itself, before any vocabulary is touched.
- **A value closes at the byte-decoded level, not the token-spelling level (T-2912, folding
  TE-368's fracture, D-SLM7593; T-2911's own token-local close blacklist is deleted as redundant).**
  The content sub-automaton carries a monotone two-mode semantic product: `U` means the decoded
  value has contributed no meaningful byte yet (every byte in `INSIGNIFICANT_VALUE_BYTES` -- the
  six JSON structural punctuation bytes `{}[],:` and JSON whitespace -- keeps the walk in `U`,
  however it is spelled: raw, a short escape, or a `\\uXXXX` escape); `M` means at least one decoded
  byte lies outside that set. Only `M` has a closing-quote edge; `U` has none. `M` is absorbing and
  keeps T-2908's own UTF-8/escape validity rules. A token that closes the value is otherwise walked
  unrestricted, at any depth, in either direction -- the close side carries none of the open side's
  risk, and a natural sentence-ending close (a period, question mark, or exclamation point plus the
  closing quote in one token) stays admitted.
- **The shipped rule carries no compound-close requirement (T-2914's Arm A was measured and
  rejected, D-SLM7596; Arm B ships, unchanged from T-2912).** A candidate requiring the closing
  quote to arrive bundled with the schema's own next required byte, in the same token, closed the
  answer-losing class this class exists to guard (0 of 97 on its own compiled artifact) but
  collapsed real-model clean-accept 10-37.5 points below every population's own forced-canonical
  baseline -- filed as the rejected candidate (`Claude/Vitruvius/t2914-probe/`), never shipped. The
  promise this leaf keeps is therefore: the value is the model's own text up to its first unescaped
  quote, not guaranteed free of every grammar-induced cut (measured cut rate 5 of 141 filed real
  decodes, 3.5%) -- a caller whose prompt's natural answer may contain an internal quote should ask
  for plain, unstructured text, or treat the returned value as possibly truncated there.

`_token_targets`, `compile_schema_to_mask_pages` and `MaskPages` are generic over whatever alphabet
the character-level DFA and trie use as dict keys; a byte value (0-255) is as valid a key as a
one-character string once every layer speaks the same alphabet.

`maxLength` on a string field, and every other JSON-Schema keyword this compiler does not
implement, are deferred capabilities: no length bound (or any other unimplemented constraint) is
added to the DFA, and a schema naming one is REJECTED at compile time rather than silently
compiling an unconstrained field and discarding the author's own stated constraint (T-2859 F2,
D-SLM7462; generalized to every keyword, T-2908 S3, Sec3.9.6).

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
from typing import Any, Callable, Mapping, Sequence

__all__ = [
    "SchemaCompileError",
    "MaskPages",
    "compile_schema_to_mask_pages",
    "zero_special_ids",
    "INSIGNIFICANT_VALUE_BYTES",
]


class SchemaCompileError(Exception):
    """G-7a's rejection: a reachable non-accepting state has an empty valid-token set.

    `state_id` names the offending compiled-DFA state; `prefix` is the canonical-
    serialization prefix (raw bytes) that reaches it; `reason` explains, in words, what
    continuation no vocabulary token can spell. `state_id` is `None` only for a structural
    rejection (a schema construct or keyword outside the D-SLM45 compilable subset) -- no
    DFA state exists yet to name in that case.
    """

    def __init__(
        self,
        message: str,
        *,
        state_id: int | None = None,
        prefix: bytes = b"",
        reason: str = "",
    ) -> None:
        super().__init__(message)
        self.state_id = state_id
        self.prefix = prefix
        self.reason = reason


# --- schema -> ordered alternation groups (D-SLM45's compilable subset) -----------------


class _StringLeaf:
    """Sentinel marking a free-text `"type": "string"` leaf's own cyclic sub-automaton
    (T-2853 design Sec3.2) in place of an ordinary alternation group of literal choices.
    `_groups()` emits a one-element group holding this sentinel for a string leaf;
    `_char_dfa()` recognizes it and builds the content/escape/`\\uXXXX` cycle directly,
    rather than the straight-line per-alternative chain every other group uses. A plain
    string equality/`is` check on the sentinel, not on `str`, is what keeps this from
    ever colliding with a real literal alternative (every real alternative is a `str`)."""

    __slots__ = ()


_STRING_LEAF = _StringLeaf()

# T-2853 Sec3.2's eight short escapes, `S_e -> S_c`, keyed by the literal character that
# follows the backslash in the token stream (not by the character it unescapes to).
_SHORT_ESCAPE_CHARS = ('"', "\\", "/", "b", "f", "n", "r", "t")
_HEX_DIGITS = "0123456789abcdefABCDEF"

# A group is ordinarily a tuple of literal alternative strings; the one exception is the
# single-element `(_STRING_LEAF,)` group a `"type": "string"` leaf emits (T-2853).
_Group = tuple[Any, ...]

# T-2912's value-level closure (Sec3.9.1, D-SLM7593): the six JSON structural punctuation
# bytes plus RFC 8259 JSON whitespace. A decoded value contributing only these bytes has not
# yet said anything meaningful and cannot close; every other byte makes the value closable.
INSIGNIFICANT_VALUE_BYTES = frozenset(b"{}[],: \t\r\n")
_INSIGNIFICANT_UESCAPE_CODEPOINTS = frozenset(INSIGNIFICANT_VALUE_BYTES)

# --- S3 (T-2908, folding TE-365 S3, Sec3.9.6): the compiler's own accepted-keyword surface,
# by branch -- derived from the keys each branch actually reads, so a schema naming any other
# key is rejected at ingestion time rather than silently compiling with the constraint unenforced.
#
# T-2917 (folding TE-370 S1, D-SLM7600): the allowlist over-rejected two shapes 1.5.0 always
# compiled. The full JSON-Schema (2020-12) annotation vocabulary -- keywords this compiler never
# enforces and were never claimed as enforced, so silently ignoring them discards nothing a
# schema author asked for -- is `title`, `description`, `default`, `examples`, `deprecated`,
# `readOnly`, `writeOnly` and `$comment` (the Meta-Data and Core "annotation" vocabularies this
# subset can reach); `$schema` and `$id` are Core identifier keywords, non-constraining for the
# same reason, accepted only at the schema root (`path == "$"`), where they are ever meaningful.
# Every one of these is accepted and ignored on every branch (or at the root); a genuinely
# constraining, unimplemented keyword (`maxLength`, `minLength`, `pattern`, `format`, `const`,
# `multipleOf`, ...) still names itself and is refused, unchanged.
_ANNOTATION_KEYWORDS: frozenset[str] = frozenset({
    "title", "description", "default", "examples", "deprecated", "readOnly", "writeOnly",
    "$comment",
})
_ROOT_ONLY_ANNOTATION_KEYWORDS: frozenset[str] = frozenset({"$schema", "$id"})

_ALLOWED_KEYWORDS: dict[str, frozenset[str]] = {
    "enum": frozenset({"enum", "type"}) | _ANNOTATION_KEYWORDS,
    "boolean": frozenset({"type"}) | _ANNOTATION_KEYWORDS,
    "string": frozenset({"type"}) | _ANNOTATION_KEYWORDS,
    "object": frozenset({"type", "properties", "required", "additionalProperties"}) | _ANNOTATION_KEYWORDS,
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

# T-2917 (TE-370 S1): `type` alongside `enum` is the pydantic/usual JSON-Schema enum form
# (every enum value already carries its own JSON type, so `type` is redundant, not
# constraining) -- accepted, and checked for agreement with every enum value's own JSON type
# rather than silently ignored, so a schema naming a `type` its own values contradict is still
# caught rather than compiling a field that can never accept what it names.
_JSON_SCHEMA_TYPE_CHECKS: dict[str, Callable[[Any], bool]] = {
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "null": lambda v: v is None,
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
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
    """`schema`'s canonical serialization as an ordered list of alternation groups.

    Each group is the set of strings the serialization may take at that position; a
    literal is a group of one. Concatenating one choice from each group, in order,
    produces exactly the canonical serializations D-SLM40 admits for this schema --
    except `_STRING_LEAF` (T-2853), which names a cyclic sub-automaton rather than a
    finite set of choices; `_char_dfa()` is what expands it.
    """
    root_extra = _ROOT_ONLY_ANNOTATION_KEYWORDS if path == "$" else frozenset()
    node_type = schema.get("type")

    if "enum" in schema:
        _reject_unimplemented_keywords(schema, path, _ALLOWED_KEYWORDS["enum"] | root_extra)
        values = schema["enum"]
        if not values:
            raise SchemaCompileError(f"empty enum at {path}", reason=f"empty enum at {path}")
        if node_type is not None:
            checker = _JSON_SCHEMA_TYPE_CHECKS.get(node_type)
            if checker is None:
                raise SchemaCompileError(
                    f"unsupported type {node_type!r} alongside enum at {path}",
                    reason=f"{node_type!r} is not one of the JSON-Schema primitive type names this compiler recognizes",
                )
            mismatched = [v for v in values if not checker(v)]
            if mismatched:
                raise SchemaCompileError(
                    f"enum value(s) {mismatched!r} do not match the declared type {node_type!r} at {path}",
                    reason="type is redundant with enum but must agree with every value when both are present",
                )
        return [tuple(json.dumps(v, ensure_ascii=False) for v in values)]

    if node_type == "boolean":
        _reject_unimplemented_keywords(schema, path, _ALLOWED_KEYWORDS["boolean"] | root_extra)
        return [("true", "false")]

    if node_type == "string":
        _reject_unimplemented_keywords(schema, path, _ALLOWED_KEYWORDS["string"] | root_extra)
        return [(_STRING_LEAF,)]

    if node_type == "object":
        _reject_unimplemented_keywords(schema, path, _ALLOWED_KEYWORDS["object"] | root_extra)
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


# --- the byte-level content sub-automaton (T-2908's alphabet, T-2912's U/M value closure) ----


def _add_string_leaf(
    edges: list[dict[int, set[int]]],
    add_node: Callable[[], int],
    cursor: int,
    content_nodes: set[int],
) -> int:
    """T-2853 design Sec3.2's cyclic sub-automaton for a `"type": "string"` leaf, wired
    from `cursor` (the position immediately before the value's opening quote) to the
    returned `following` node (immediately after the closing quote) -- the identical
    cursor->following convention every other group already uses, so the leaf composes
    into the existing group-chain model rather than replacing it.

    Two parallel content regions, both fed by T-2908's byte-level UTF-8-validity
    construction, carrying T-2912's own monotone semantic product (Sec3.9.1):

    - `M` ("meaningful"): T-2908's original content automaton, unchanged -- an 11-state
      byte-level UTF-8-validity sub-automaton (one lead-byte state per length class, with
      continuation ranges narrowed to exclude overlong encodings, surrogates, and
      codepoints past U+10FFFF) plus the escape branch and the `\\uXXXX` hex chain. `M` is
      absorbing and is the only region with a closing-quote edge.
    - `U` ("uncommitted"): entered directly from the opening quote. Every ASCII byte that
      decodes to one of `INSIGNIFICANT_VALUE_BYTES` (raw, via a short escape, or via a
      `\\uXXXX` escape) keeps the walk in `U`; every other byte -- including the first byte
      of any multi-byte UTF-8 sequence, since a non-ASCII scalar is meaningful under this
      byte language as soon as its valid lead arrives -- transitions into `M`. `U` has no
      closing-quote edge: a value that has contributed no meaningful byte cannot close.

    Every node this function creates is recorded in `content_nodes` (T-2910) so `_char_dfa`
    can label which DFA states belong to the content region once subset construction has
    run -- `_token_targets`'s open-side boundary discipline consults this. `following` (the
    literal-skeleton state resumed after the close) is NOT a content node.
    """
    quote, backslash = 0x22, 0x5C

    def node() -> int:
        result = add_node()
        content_nodes.add(result)
        return result

    def edge(state: int, lo: int, hi: int, target: int) -> None:
        for byte_val in range(lo, hi + 1):
            edges[state].setdefault(byte_val, set()).add(target)

    # --- M: T-2908's own UTF-8-valid content automaton, absorbing, the only region with a
    # closing-quote edge. ---
    m_sc = node()
    m_lead2 = node()
    edge(m_sc, 0xC2, 0xDF, m_lead2)
    edge(m_lead2, 0x80, 0xBF, m_sc)

    m_lead3a, m_lead3b, m_lead3c, m_mid3 = node(), node(), node(), node()
    edge(m_sc, 0xE0, 0xE0, m_lead3a)
    edge(m_sc, 0xE1, 0xEC, m_lead3b)
    edge(m_sc, 0xED, 0xED, m_lead3c)
    edge(m_sc, 0xEE, 0xEF, m_lead3b)
    edge(m_lead3a, 0xA0, 0xBF, m_mid3)  # excludes the overlong 0x80-0x9F continuation
    edge(m_lead3b, 0x80, 0xBF, m_mid3)
    edge(m_lead3c, 0x80, 0x9F, m_mid3)  # excludes the surrogate range 0xA0-0xBF (U+D800-DFFF)
    edge(m_mid3, 0x80, 0xBF, m_sc)

    m_lead4a, m_lead4b, m_lead4c, m_mid4a, m_mid4b = node(), node(), node(), node(), node()
    edge(m_sc, 0xF0, 0xF0, m_lead4a)
    edge(m_sc, 0xF1, 0xF3, m_lead4b)
    edge(m_sc, 0xF4, 0xF4, m_lead4c)
    edge(m_lead4a, 0x90, 0xBF, m_mid4a)  # excludes the overlong 0x80-0x8F continuation
    edge(m_lead4b, 0x80, 0xBF, m_mid4a)
    edge(m_lead4c, 0x80, 0x8F, m_mid4a)  # excludes codepoints past U+10FFFF
    edge(m_mid4a, 0x80, 0xBF, m_mid4b)
    edge(m_mid4b, 0x80, 0xBF, m_sc)
    # 0xF5-0xFF: no edge (no valid codepoint starts with these bytes).

    for byte_val in range(0x20, 0x80):
        if byte_val not in (quote, backslash):
            edges[m_sc].setdefault(byte_val, set()).add(m_sc)

    m_escape = node()
    edges[m_sc].setdefault(backslash, set()).add(m_escape)
    for escape_char in _SHORT_ESCAPE_CHARS:
        edges[m_escape].setdefault(ord(escape_char), set()).add(m_sc)
    m_hex = [node() for _ in range(4)]
    edges[m_escape].setdefault(ord("u"), set()).add(m_hex[0])
    for index, state in enumerate(m_hex):
        target = m_hex[index + 1] if index + 1 < len(m_hex) else m_sc
        for digit in _HEX_DIGITS:
            edges[state].setdefault(ord(digit), set()).add(target)

    # --- U: entered on the opening quote; has no closing-quote edge. ---
    u_sc = node()
    edges[cursor].setdefault(quote, set()).add(u_sc)
    for byte_val in range(0x20, 0x80):
        if byte_val in (quote, backslash):
            continue
        target = u_sc if byte_val in INSIGNIFICANT_VALUE_BYTES else m_sc
        edges[u_sc].setdefault(byte_val, set()).add(target)

    # A non-ASCII scalar is meaningful under the byte language as soon as its valid lead
    # arrives -- every multi-byte UTF-8 sequence routes into M's own lead states.
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
            for digit in _HEX_DIGITS:
                edges[state].setdefault(ord(digit), set()).add(target)
        return generic_remaining[remaining]

    prefix_nodes: dict[str, int] = {}

    def unicode_prefix(prefix: str) -> int:
        if prefix in prefix_nodes:
            return prefix_nodes[prefix]
        state = node()
        prefix_nodes[prefix] = state
        for digit in _HEX_DIGITS:
            normalized = digit.lower()
            candidate = prefix + normalized
            if len(candidate) == 4:
                target = u_sc if int(candidate, 16) in _INSIGNIFICANT_UESCAPE_CODEPOINTS else m_sc
            elif any(value.startswith(candidate) for value in insignificant_hex):
                target = unicode_prefix(candidate)
            else:
                target = significant_remainder(4 - len(candidate))
            edges[state].setdefault(ord(digit), set()).add(target)
        return state

    edges[u_escape].setdefault(ord("u"), set()).add(unicode_prefix(""))

    following = add_node()  # literal skeleton; deliberately not a content node
    edges[m_sc].setdefault(quote, set()).add(following)
    return following


def _char_dfa(groups: Sequence[_Group]) -> tuple[list[dict[int, int]], int, frozenset[int], frozenset[int]]:
    """Returns `(dfa, start, accepting, content_states)`. `content_states` (T-2910) is the
    DFA state ids whose underlying NFA node set intersects the string leaf's own content
    nodes -- `_token_targets` consults it to tell content from literal skeleton."""
    edges: list[dict[int, set[int]]] = [{}]
    content_nodes: set[int] = set()

    def add_node() -> int:
        edges.append({})
        return len(edges) - 1

    cursor = 0
    for group in groups:
        if len(group) == 1 and group[0] is _STRING_LEAF:
            cursor = _add_string_leaf(edges, add_node, cursor, content_nodes)
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
    content_states: set[int] = set()
    queue: deque[frozenset[int]] = deque([start_set])
    while queue:
        current = queue.popleft()
        state = ids[current]
        if accept_node in current:
            accepting.add(state)
        if current & content_nodes:
            content_states.add(state)
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
    return dfa, 0, frozenset(accepting), frozenset(content_states)


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


# --- the vocabulary trie (drives the token-level DFA construction) ------------------------


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
        if node.token_id is None:  # first (lowest-index) token spelling this piece wins
            node.token_id = token_id
    return root


def zero_special_ids(vocab: Sequence[bytes], special_ids: frozenset[int]) -> list[bytes]:
    """T-2910 (folding TE-366 group 2): zeroes every special/added tokenizer id's bytes so
    `_vocab_trie`'s existing empty-piece guard (`if not piece: continue`) excludes it --
    structurally, for every state of every schema this compiler ever compiles, not only the
    string leaf's own content states. T-2917 (folding TE-370 C1, D-SLM7600):
    `compile_schema_to_mask_pages` applies this itself, to its own copy of `vocab`, from its
    own required `special_ids` argument -- exposed as its own public function too, for a
    reference/probe module that wants the zeroed vocabulary directly without compiling
    anything."""
    out = list(vocab)
    for i in special_ids:
        if 0 <= i < len(out):
            out[i] = b""
    return out


def _token_targets(
    dfa: Sequence[Mapping[int, int]],
    state: int,
    root: _TrieNode,
    content_states: frozenset[int],
) -> dict[int, int]:
    """Every vocabulary token drivable from `state`, and the character-DFA state each
    reaches -- T-2910's asymmetric open/close boundary discipline (D-SLM7589), re-derived
    after a symmetric first attempt regressed clean-accept on real prompts (Sec3.9.1).

    OPENING a value (a token transitioning from a non-content state to a content state) is
    restricted to the token's own first byte: an opening crossing past depth 0 is refused
    outright, not merely stopped, so a vocabulary token can never carry pre-value structure
    and content in one piece.

    CLOSING a value (content to non-content) is walked unrestricted, at any depth -- the
    close side carries none of the open side's risk (TE-366's own census attributed every
    boundary-leakage instance to an open-side compound token, none to a close-side one), and
    restricting it symmetrically cost the walk its own natural stopping point on real
    prompts without closing any defect TE-366 found. T-2912's own value-level `U`/`M`
    product (`_add_string_leaf`) is what keeps a close from admitting an insignificant-only
    value; this function no longer carries T-2911's own token-local structural-close
    blacklist, deleted as redundant (V3/V4, Sec3.9.3: it narrows spellings, not values, and
    is not part of the accepted-language proof).

    A token that never crosses the boundary -- wholly inside content (free multi-byte answer
    tokens, at full width) or wholly outside it (compound literal-skeleton tokens, per the
    design's own existing harmless-crossing precedent) -- is walked exactly as before, on
    both sides.
    """
    targets: dict[int, int] = {}
    # stack entries: (trie_node, dfa_state, depth, stop_after -- True only for a node reached
    # by an OPENING crossing, which may only happen at depth 0 and forbids further descent).
    stack: list[tuple[_TrieNode, int, int, bool]] = [(root, state, 0, False)]
    while stack:
        node, current, depth, stop_after = stack.pop()
        if node.token_id is not None:
            targets[node.token_id] = current
        if stop_after:
            continue  # an opening token contributes no further bytes
        row = dfa[current]
        cur_is_content = current in content_states
        if len(row) < len(node.children):
            pairs = ((byte_val, node.children.get(byte_val), target)
                     for byte_val, target in row.items())
        else:
            pairs = ((byte_val, child, row.get(byte_val))
                     for byte_val, child in node.children.items())
        for byte_val, child, target in pairs:
            if child is None or target is None:
                continue
            opening = (not cur_is_content) and target in content_states
            if opening and depth != 0:
                continue  # opening past the token's first byte is refused outright
            # a CLOSING crossing and any non-crossing transition both continue unrestricted.
            stack.append((child, target, depth + 1, opening))
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
        vocab: Sequence[bytes],
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

    def accepts(self, s: bytes) -> bool:
        """G5-1 fuzz-suite convenience: does `s` walk the compiled DFA, token by token
        (greedy longest-match over each state's own valid-token row), to an accepting
        state with no leftover input? Not a runtime primitive -- the runtime never
        tokenizes a raw byte string against the mask; it consults `page(state)` against
        logits the model already produced. This walk exists so the fuzz suite can
        assert admitted/non-admitted byte strings without a full decode loop."""
        state = self.start
        pos = 0
        n = len(s)
        while pos < n:
            row = self.transitions.get(state, {})
            best: tuple[bytes, int] | None = None
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


def compile_schema_to_mask_pages(
    schema: Mapping[str, Any],
    vocab: Sequence[bytes],
    *,
    special_ids: frozenset[int],
) -> MaskPages:
    """Compile `schema` (D-SLM45's per-field subset) to a token-level DFA with per-state
    valid-token bitmask pages, over `vocab` (each token's own undecoded bytes).

    `special_ids` (T-2917, folding TE-370 C1, D-SLM7600) is required and keyword-only: this
    function applies `zero_special_ids(vocab, special_ids)` to its own copy of `vocab` before
    the trie is built, so every id in `special_ids` is excluded from every state of every
    schema, structurally -- a caller cannot reach an admitting compile by forgetting the step
    (pass `special_ids=frozenset()` to compile deliberately without excluding any id).

    Raises `TypeError` if any `vocab` element is not `bytes` -- this compiler moved from
    `Sequence[str]` to `Sequence[bytes]` at T-2908; a caller still constructing a `str`
    vocabulary must encode each piece first (`tools/convert_tokenizer.py`'s own
    `TokenizerTables.id_to_bytes`, or `str.encode`).

    Raises `SchemaCompileError` -- naming the offending state, the serialization prefix
    that reaches it, and the continuation no vocabulary token can spell -- for any
    reachable non-accepting state with an empty valid-token set (G-7a). This is the
    compiler's own proof obligation: an all-zero mask is indistinguishable from a
    legitimate one at the runtime's AND step (D-SLM40), so the guarantee must be
    structural, here, at compile time -- the runtime must not (and, by this design, does
    not) check for it.
    """
    for index, piece in enumerate(vocab):
        if not isinstance(piece, bytes):
            raise TypeError(
                f"vocab[{index}] is {type(piece).__name__!r}, not bytes -- this compiler takes "
                "Sequence[bytes] (each token's own undecoded bytes), not Sequence[str]. Encode "
                "each vocabulary piece to bytes before calling (T-2908's byte-level port)."
            )
    vocab = tuple(zero_special_ids(vocab, special_ids))
    char_dfa, char_start, char_accepting, content_states = _char_dfa(_groups(schema, "$"))
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
        targets = _token_targets(char_dfa, current, trie, content_states)
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

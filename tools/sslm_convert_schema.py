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

A `"type": "array"` node (T-3094 S1, plan of record `Claude/Plans/superslm-multicommand-array/plan.md`
Sec4.1, Wizard repo) compiles to a cyclic item automaton:

- **Grammar.** `[`, then the `minItems` required item occurrences separated by `,`; after the minimum the
  automaton offers `]` or `,` followed by the same item automaton again. `minItems: 0` has a direct `[]`
  path, a first-item path and a repeat-item path. The loop edge starts only after a complete item, so a
  trailing, leading or doubled comma is never accepted. `items` must be one schema object; arrays nest by
  the same recursion (an array's item may hold another array).
- **Construction.** `_groups()` emits an array as a one-element group holding an `_ArrayGroup`
  sentinel; `_char_dfa()` wires the byte NFA recursively (epsilon-free) and determinizes it by the same
  subset construction every schema already uses. `_token_targets`, the special-id exclusion and the
  `MaskPages` representation are unchanged; a schema without an array compiles to the same table as
  before.
- **Refused.** `maxItems` is refused by name even where a small value could be flattened (Sec4.1).
- **Resource admission.** Seven limits (`RELEASE_LIMITS`, the frozen provisional release values of the
  T-3094 S0 limits freeze) are checked at their counting stage, each with its own named diagnostic
  (`LIMIT_DIAGNOSTICS`): `minItems` per array (the diagnostic names the array's RFC 6901 schema
  pointer) and nesting depth before any automaton is built; NFA states as each node is created; token-DFA
  states, token transitions, SCM1 section bytes and tracked temporary bytes as the token table grows. The
  tracked budget is reserved before each charged structure is allocated, so a refusal stops at the
  crossing rather than after the whole table exists. `limits=` replaces named limits for one call only;
  the compiled artifact reports every count in `MaskPages.resource_counts`.
- **Every reachable state can finish.** The empty-row refusal (G-7a) is the zero-step case of the class
  rule this compiler enforces: a schema whose token DFA has a reachable state from which no accepting
  state is reachable is refused by name, with no artifact (a cyclic table can loop forever with every row
  non-empty when the vocabulary cannot spell an exit).

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
from collections.abc import Hashable
from typing import Any, Callable, Mapping, Sequence

__all__ = [
    "SchemaCompileError",
    "MaskPages",
    "compile_schema_to_mask_pages",
    "zero_special_ids",
    "serialize_scm1",
    "fnv1a64",
    "INSIGNIFICANT_VALUE_BYTES",
    "RELEASE_LIMITS",
    "LIMIT_DIAGNOSTICS",
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


class _ArrayGroup:
    """Sentinel group for a `"type": "array"` node (T-3094 S1, plan Sec4.1): `_groups()` emits a
    one-element group holding one instance, exactly as a string leaf emits `(_STRING_LEAF,)`, and
    `_char_dfa()` wires it as the cyclic item automaton (`_wire_array`). `item` is the item
    schema's own ordered group list; `min_items` is the array's own required minimum."""

    __slots__ = ("min_items", "item")

    def __init__(self, min_items: int, item: list) -> None:
        self.min_items = min_items
        self.item = item

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
    "array": frozenset({"type", "items", "minItems"}) | _ANNOTATION_KEYWORDS,
}

_KEYWORD_REASON_OVERRIDES: dict[str, str] = {
    "maxItems": (
        "the production converter refuses maxItems even where a small value could be flattened "
        "into positional copies: the array compiles to one cyclic item automaton with no upper "
        "bound, so accepting the keyword would compile an unbounded array while discarding the "
        "author's stated bound (T-3094 plan Sec4.1)"
    ),
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
#
# T-2920 (TE-372 S2): JSON Schema (and OpenAPI 3.1) also allow `type` to be a LIST of type
# names -- e.g. `["string", "null"]` for a nullable string -- meaning a value is valid if it
# matches ANY ONE of the listed types (the spec's own union/anyOf reading of a type array).
# `_groups()` normalizes a scalar `type` into a one-element list and a list `type` into itself,
# then checks each enum value against every listed type's checker with `any(...)` -- the direct
# generalization of the scalar rule immediately above (mismatch raises SchemaCompileError) to a
# union of types rather than a single one, so a value matching none of the listed types is
# refused by the identical mechanism a single mismatched scalar type already used, and a value
# matching at least one is accepted exactly as the scalar case already accepts a match.
_JSON_SCHEMA_TYPE_CHECKS: dict[str, Callable[[Any], bool]] = {
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "null": lambda v: v is None,
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


def _reject_unimplemented_keywords(
    schema: Mapping[str, Any], path: str, allowed: frozenset[str], pointer: str | None = None
) -> None:
    where = path if pointer is None else f"{path} (schema pointer {pointer})"
    for key in sorted(schema):
        if key in allowed:
            continue
        reason = _KEYWORD_REASON_OVERRIDES.get(key, (
            f"{key!r} is a JSON-Schema keyword this compiler does not implement; a schema "
            f"naming it is rejected rather than silently accepted with the constraint "
            f"unenforced (generalizes the maxLength precedent, T-2859 F2/S3, to every "
            f"keyword outside {sorted(allowed)!r})"
        ))
        raise SchemaCompileError(f"{key} is not supported at {where}", reason=reason)


def _pointer_token(key: str) -> str:
    """One RFC 6901 reference token: `~` is written `~0` and `/` is written `~1`."""
    return key.replace("~", "~0").replace("/", "~1")


def _groups(
    schema: Mapping[str, Any],
    path: str,
    *,
    pointer: str = "",
    limits: Mapping[str, int] | None = None,
    arrays: dict[str, int] | None = None,
) -> list[_Group]:
    """`schema`'s canonical serialization as an ordered list of alternation groups.

    Each group is the set of strings the serialization may take at that position; a
    literal is a group of one. Concatenating one choice from each group, in order,
    produces exactly the canonical serializations D-SLM40 admits for this schema --
    except `_STRING_LEAF` (T-2853), which names a cyclic sub-automaton rather than a
    finite set of choices, and an `_ArrayGroup` (T-3094 S1), which names the cyclic
    item automaton; `_char_dfa()` is what expands both.

    `pointer` is the node's RFC 6901 pointer into the schema document (array diagnostics name
    it). `arrays`, when given, collects each array's pointer and `minItems`; `limits`, when
    given, applies the per-array `minItems` limit (T-3094 S1, D-SLM8087: every array
    independently).
    """
    root_extra = _ROOT_ONLY_ANNOTATION_KEYWORDS if path == "$" else frozenset()
    node_type = schema.get("type")

    if "enum" in schema:
        if node_type in ("object", "array") and set(schema) - (_ALLOWED_KEYWORDS["enum"] | root_extra):
            # An object or array node carrying `enum`: the node is refused either way (its own
            # structural keys are outside the enum allowlist), and the keyword this compiler does
            # not implement at that node is `enum` itself, so the refusal names it.
            raise SchemaCompileError(
                f"enum is not supported at {path} on a {node_type} node",
                reason=f"enum is implemented only on a scalar leaf; a {node_type} node's language "
                       f"comes from its own structural keywords",
            )
        _reject_unimplemented_keywords(schema, path, _ALLOWED_KEYWORDS["enum"] | root_extra)
        values = schema["enum"]
        if not values:
            raise SchemaCompileError(f"empty enum at {path}", reason=f"empty enum at {path}")
        if node_type is not None:
            node_types = node_type if isinstance(node_type, list) else [node_type]
            if not node_types:
                raise SchemaCompileError(
                    f"empty type list alongside enum at {path}",
                    reason=f"a type list at {path} names no candidate type, so no enum value could ever satisfy it",
                )
            checkers: list[Callable[[Any], bool]] = []
            for t in node_types:
                checker = _JSON_SCHEMA_TYPE_CHECKS.get(t) if isinstance(t, Hashable) else None
                if checker is None:
                    raise SchemaCompileError(
                        f"unsupported type {t!r} alongside enum at {path}",
                        reason=f"{t!r} is not one of the JSON-Schema primitive type names this compiler recognizes",
                    )
                checkers.append(checker)
            mismatched = [v for v in values if not any(checker(v) for checker in checkers)]
            if mismatched:
                raise SchemaCompileError(
                    f"enum value(s) {mismatched!r} do not match the declared type {node_type!r} at {path}",
                    reason=(
                        "type is redundant with enum but must agree with every value when both "
                        "are present; a type list is a union, so agreement means matching at "
                        "least one listed type"
                    ),
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
                f"object at {path} permits additional properties (additionalProperties must be false)",
                reason="additionalProperties must be false; an open object has no fixed serialization",
            )
        if required != list(properties):
            raise SchemaCompileError(
                f"object at {path} has optional or reordered keys (required must list every property, in properties order)",
                reason="every property must be required, in properties order (D-SLM45); optional/reordered keys are outside the compilable subset",
            )
        groups: list[_Group] = [("{",)]
        for index, (key, sub_schema) in enumerate(properties.items()):
            if index:
                groups.append((",",))
            groups.append((json.dumps(key, ensure_ascii=False) + ":",))
            groups.extend(_groups(sub_schema, f"{path}.{key}",
                                  pointer=f"{pointer}/properties/{_pointer_token(key)}",
                                  limits=limits, arrays=arrays))
        groups.append(("}",))
        return groups

    if node_type == "array":
        return [(_array_group(schema, path, pointer, root_extra, limits, arrays),)]

    raise SchemaCompileError(
        f"unsupported construct at {path}",
        reason=f"type {node_type!r} is outside D-SLM45's compilable subset (objects with known keys, arrays of one item schema, enums, booleans, unbounded free-text strings; cross-field constraints are scored, not compiled)",
    )


def _array_group(
    schema: Mapping[str, Any],
    path: str,
    pointer: str,
    root_extra: frozenset[str],
    limits: Mapping[str, int] | None,
    arrays: dict[str, int] | None,
) -> _ArrayGroup:
    """An array node's `_ArrayGroup` (T-3094 S1, plan Sec4.1). Every refusal here names the
    array's RFC 6901 schema pointer."""
    where = f"{path} (schema pointer {pointer})"
    _reject_unimplemented_keywords(schema, path, _ALLOWED_KEYWORDS["array"] | root_extra, pointer)
    if "items" not in schema:
        raise SchemaCompileError(
            f"array at {where} has no items schema",
            reason="items must name the one schema every array item follows",
        )
    items = schema["items"]
    if not isinstance(items, Mapping):
        raise SchemaCompileError(
            f"items at {where} must be a single schema object, not {type(items).__name__}",
            reason="tuple-form (positional) items and boolean items are outside the compilable "
                   "subset; every item follows one schema",
        )
    min_items = schema.get("minItems", 0)
    if isinstance(min_items, bool) or not isinstance(min_items, int) or min_items < 0:
        raise SchemaCompileError(
            f"minItems at {where} must be a non-negative integer, not {min_items!r}",
            reason="minItems is the array's required minimum item count",
        )
    if arrays is not None:
        arrays[pointer] = min_items
    if limits is not None and min_items > limits["minItems"]:
        raise SchemaCompileError(
            f"{LIMIT_DIAGNOSTICS['minItems']} exceeded: the array at schema pointer {pointer} "
            f"requires minItems {min_items}, above the per-array limit of {limits['minItems']}",
            reason="each required item is a separate copy of the item automaton; the limit applies "
                   "to every array independently (D-SLM8087)",
        )
    return _ArrayGroup(min_items, _groups(items, f"{path}[]", pointer=f"{pointer}/items",
                                          limits=limits, arrays=arrays))


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


class _NFA:
    """The epsilon-free byte NFA under construction: `edges[node][byte]` is the node's target set.
    Node 0 is the start. Every node is created through `add_node`, which charges the admission
    budget (when there is one) before the node exists; edges are charged as they are added."""

    def __init__(self, admission: "_Admission | None") -> None:
        self.admission = admission
        self.edges: list[dict[int, set[int]]] = []
        self.content_nodes: set[int] = set()
        self.add_node()

    def add_node(self) -> int:
        if self.admission is not None:
            self.admission.nfa_node()
        self.edges.append({})
        return len(self.edges) - 1

    def edge_count(self, nodes: Sequence[int] | range) -> int:
        return sum(len(targets) for node in nodes for targets in self.edges[node].values())

    def charge_edges(self, count: int) -> None:
        if self.admission is not None and count:
            self.admission.nfa_edges(count)

    def literal(self, start: int, raw: bytes, end: int | None = None) -> int:
        """A literal byte chain from `start` to `end` (a fresh node when `end` is None)."""
        following = self.add_node() if end is None else end
        node = start
        added = 0
        for index, byte_val in enumerate(raw):
            target = following if index == len(raw) - 1 else self.add_node()
            targets = self.edges[node].setdefault(byte_val, set())
            if target not in targets:
                targets.add(target)
                added += 1
            node = target
        self.charge_edges(added)
        return following


def _wire(nfa: _NFA, groups: Sequence[_Group], cursor: int) -> int:
    """Wires `groups` into `nfa` from `cursor`; returns the node after the last group."""
    for group in groups:
        if len(group) == 1 and group[0] is _STRING_LEAF:
            before = nfa.edge_count((cursor,))
            first_new = len(nfa.edges)
            following = _add_string_leaf(nfa.edges, nfa.add_node, cursor, nfa.content_nodes)
            nfa.charge_edges(nfa.edge_count((cursor,)) - before + nfa.edge_count(range(first_new, len(nfa.edges))))
            cursor = following
            continue
        if len(group) == 1 and isinstance(group[0], _ArrayGroup):
            cursor = _wire_array(nfa, group[0], cursor)
            continue
        following = nfa.add_node()
        for alternative in group:
            if not alternative:
                raise SchemaCompileError(
                    "an alternation contains the empty string",
                    reason="a zero-length alternative makes the serialization ambiguous",
                )
            nfa.literal(cursor, alternative.encode("utf-8"), following)
        cursor = following
    return cursor


def _wire_array(nfa: _NFA, array: _ArrayGroup, cursor: int) -> int:
    """The cyclic item automaton (T-3094 S1, plan Sec4.1): `[`, the `min_items` required item
    occurrences separated by `,`, then `]` or `,` back into the same (last) item copy. The loop
    edge leaves only a complete item, so no comma is ever followed by `]`. `minItems: 0` has the
    direct `[]` path, a first-item copy and a repeat-item copy whose `,` loops into itself (the
    first copy's start also carries `]`, so it cannot be the loop target)."""
    following = nfa.add_node()
    cur = nfa.literal(cursor, b"[")
    if array.min_items == 0:
        nfa.literal(cur, b"]", following)
        first_end = _wire(nfa, array.item, cur)
        nfa.literal(first_end, b"]", following)
        repeat = nfa.add_node()
        nfa.literal(first_end, b",", repeat)
        repeat_end = _wire(nfa, array.item, repeat)
        nfa.literal(repeat_end, b"]", following)
        nfa.literal(repeat_end, b",", repeat)
        return following
    repeat = cur
    for index in range(array.min_items):
        if index:
            cur = nfa.literal(cur, b",")
        repeat = cur
        cur = _wire(nfa, array.item, cur)
    nfa.literal(cur, b"]", following)
    nfa.literal(cur, b",", repeat)
    return following


def _char_dfa(
    groups: Sequence[_Group], admission: "_Admission | None" = None
) -> tuple[list[dict[int, int]], int, frozenset[int], frozenset[int]]:
    """Returns `(dfa, start, accepting, content_states)`. `content_states` (T-2910) is the
    DFA state ids whose underlying NFA node set intersects the string leaf's own content
    nodes -- `_token_targets` consults it to tell content from literal skeleton.

    With `admission` (T-3094 S1), every NFA node is admitted against the NFA state limit as it
    is created, and every NFA edge and byte-DFA state is reserved against the tracked budget
    before it is stored."""
    nfa = _NFA(admission)
    accept_node = _wire(nfa, groups, 0)
    edges, content_nodes = nfa.edges, nfa.content_nodes

    start_set = frozenset({0})
    if admission is not None:
        admission.byte_dfa_state()
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
                if admission is not None:
                    admission.byte_dfa_state()
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
        resource_counts: Mapping[str, Any] | None = None,
    ) -> None:
        self.vocab = tuple(vocab)
        self.transitions: dict[int, dict[int, int]] = {s: dict(row) for s, row in transitions.items()}
        self.start = start
        self.accepting = frozenset(accepting)
        # T-3094 S1: the seven admitted counts of the compile that produced this table (`minItems`
        # maps each array's RFC 6901 schema pointer to its minimum). None for a hand-built table.
        self.resource_counts: dict[str, Any] | None = (
            None if resource_counts is None else dict(resource_counts))

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


# --- resource admission (T-3094 S1, plan Sec4.1 resource admission contract) ---------------------

# The frozen provisional release limits (T-3103 S0 limits freeze, D-SLM8087 rule, D-SLM8095 values;
# keys use the T-3094 names). A count equal to its limit is admitted; a count above it is refused.
RELEASE_LIMITS: dict[str, int] = {
    "minItems": 8,                              # per array, every array independently
    "recursion_depth": 8,                       # a leaf is 1; an object or array is 1 + its deepest child
    "nfa_states": 4_096,                        # byte NFA nodes, the start node included
    "dfa_states": 4_096,                        # token-DFA states (one SCM1 mask page each)
    "token_transitions": 8_388_608,             # (state, token) entries in the token table
    "serialized_mask_bytes": 134_217_728,       # this schema's SCM1 section, name included
    "tracked_temporary_bytes": 268_435_456,     # the compiler's charged allocations (below)
}

LIMIT_DIAGNOSTICS: dict[str, str] = {
    "minItems": "minItems limit",
    "recursion_depth": "recursion depth limit",
    "nfa_states": "NFA state limit",
    "dfa_states": "DFA state limit",
    "token_transitions": "token transition limit",
    "serialized_mask_bytes": "serialized mask byte limit",
    "tracked_temporary_bytes": "tracked temporary memory limit",
}

# SCM1 layout (t2119 design Sec13.2, as `serialize_scm1` writes it): a 24-byte header, one 56-byte
# descriptor per schema, the name blob, then per schema the mask pages, the (states + 1) CSR state
# offsets, the accepting ids and the 8-byte (token, target) transitions.
_SCM1_HEADER_BYTES = 24
_SCM1_DESCRIPTOR_BYTES = 56

# Tracked-allocation charges, per structure, of the frozen allocation model the limits were derived
# from (T-3094 S0 `resource_pairs.py`, plan Sec4.1 memory paragraph): an NFA node, an NFA byte edge,
# a DFA state (charged for every byte-DFA state and every token-DFA state), the vocabulary trie's
# root table, a token transition's staging entry, plus the SCM1 section bytes the table will occupy.
_CHARGE_NFA_NODE = 64
_CHARGE_NFA_EDGE = 16
_CHARGE_DFA_STATE = 48
_CHARGE_TRIE = 257 * 32
_CHARGE_TRANSITION = 16


def _resolve_limits(limits: Mapping[str, int] | None) -> dict[str, int]:
    """A fresh per-call copy of `RELEASE_LIMITS` with `limits` replacing only the keys it names
    (the test harness and the attachment command inject smaller limits through the same guard).
    Nothing persists past the call that received it."""
    active = dict(RELEASE_LIMITS)
    for key, value in (limits or {}).items():
        if key not in active:
            raise ValueError(f"unknown resource limit {key!r}; the limits are {sorted(active)}")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"resource limit {key!r} must be a non-negative integer, not {value!r}")
        active[key] = value
    return active


def _schema_depth(schema: Mapping[str, Any], limit: int) -> int:
    """The schema's nesting depth (a leaf is 1; an object or array is 1 + its deepest child),
    refused by name as soon as any node lies deeper than `limit` -- before anything is built and
    without descending further, so a hostile nesting cannot exhaust the walk."""
    deepest = 0
    stack: list[tuple[Any, int, str]] = [(schema, 1, "$")]
    while stack:
        node, level, path = stack.pop()
        if level > limit:
            raise SchemaCompileError(
                f"{LIMIT_DIAGNOSTICS['recursion_depth']} exceeded: the schema nests {level} levels "
                f"deep at {path}, above the limit of {limit}",
                reason="nesting depth is checked before any automaton is built",
            )
        deepest = max(deepest, level)
        if not isinstance(node, Mapping):
            continue
        if node.get("type") == "object" and isinstance(node.get("properties"), Mapping):
            for key, child in node["properties"].items():
                stack.append((child, level + 1, f"{path}.{key}"))
        elif node.get("type") == "array" and isinstance(node.get("items"), Mapping):
            stack.append((node["items"], level + 1, f"{path}[]"))
    return deepest


class _Admission:
    """One compile's running resource counts, each checked against its limit at the moment its
    structure is about to be created -- never after the table exists. `tracked` accumulates the
    charges above and is reserved before the charged structure is stored."""

    def __init__(self, limits: Mapping[str, int], vocab_size: int) -> None:
        self.limits = limits
        self.page_bytes = (vocab_size + 7) // 8
        self.counts = {"nfa_states": 0, "dfa_states": 0, "token_transitions": 0,
                       "serialized_mask_bytes": 0, "tracked_temporary_bytes": 0}

    def _admit(self, key: str, amount: int, what: str) -> None:
        value = self.counts[key] + amount
        limit = self.limits[key]
        if value > limit:
            raise SchemaCompileError(
                f"{LIMIT_DIAGNOSTICS[key]} exceeded: {what} brings {key} to {value:,}, "
                f"above the limit of {limit:,}",
                reason=f"refused at the crossing, before {what} was created",
            )
        self.counts[key] = value

    def _reserve(self, nbytes: int, what: str) -> None:
        self._admit("tracked_temporary_bytes", nbytes, what)

    def nfa_node(self) -> None:
        self._admit("nfa_states", 1, "one more byte-NFA node")
        self._reserve(_CHARGE_NFA_NODE, "one more byte-NFA node")

    def nfa_edges(self, count: int) -> None:
        self._reserve(count * _CHARGE_NFA_EDGE, f"{count} byte-NFA edge(s)")

    def byte_dfa_state(self) -> None:
        self._reserve(_CHARGE_DFA_STATE, "one more byte-DFA subset state")

    def trie(self) -> None:
        self._reserve(_CHARGE_TRIE, "the vocabulary trie")

    def section_header(self, name_bytes: int) -> None:
        fixed = _SCM1_HEADER_BYTES + _SCM1_DESCRIPTOR_BYTES + name_bytes + 4  # + the closing CSR offset
        self._admit("serialized_mask_bytes", fixed, "the SCM1 header, descriptor and name")
        self._reserve(fixed, "the SCM1 header, descriptor and name")

    def token_state(self) -> None:
        self._admit("dfa_states", 1, "one more token-DFA state")
        per_state = self.page_bytes + 4  # its mask page and its CSR offset
        self._admit("serialized_mask_bytes", per_state, "one more mask page and CSR offset")
        self._reserve(_CHARGE_DFA_STATE + per_state, "one more token-DFA state and its mask page")

    def accepting_state(self) -> None:
        self._admit("serialized_mask_bytes", 4, "one more accepting-state id")
        self._reserve(4, "one more accepting-state id")

    def transitions(self, count: int) -> None:
        self._admit("token_transitions", count, f"a row of {count:,} token transition(s)")
        self._admit("serialized_mask_bytes", 8 * count, f"a row of {count:,} serialized transition(s)")
        self._reserve((_CHARGE_TRANSITION + 8) * count, f"a row of {count:,} transition(s) and its serialized form")


def _states_that_cannot_finish(transitions: Mapping[int, Mapping[int, int]], accepting: set[int]) -> list[int]:
    """Every state of the (fully reachable) token table from which no accepting state is reachable."""
    back: dict[int, set[int]] = {}
    for state, row in transitions.items():
        for target in row.values():
            back.setdefault(target, set()).add(state)
    live = set(accepting)
    frontier = list(accepting)
    while frontier:
        for source in back.get(frontier.pop(), ()):
            if source not in live:
                live.add(source)
                frontier.append(source)
    return sorted(s for s in transitions if s not in live)


# --- the compiler ---------------------------------------------------------------------------


def compile_schema_to_mask_pages(
    schema: Mapping[str, Any],
    vocab: Sequence[bytes],
    *,
    special_ids: frozenset[int],
    limits: Mapping[str, int] | None = None,
    name: str | bytes | None = None,
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
    not) check for it. The same refusal covers its general case (T-3094 S1): a reachable
    state with a non-empty row from which no accepting state can be reached (a cyclic table
    whose exit no token spells) is refused by name, with no artifact.

    `limits` (T-3094 S1) replaces the named `RELEASE_LIMITS` for this call only; a count above
    its limit raises `SchemaCompileError` whose message starts with that limit's
    `LIMIT_DIAGNOSTICS` string. `name` is the SCM1 schema name this table will be attached
    under; the section-byte count includes it (an omitted name counts as zero bytes, so the
    attachment command, which always names the schema, is where that count is exact). The
    returned `MaskPages.resource_counts` holds every count this compile admitted.
    """
    for index, piece in enumerate(vocab):
        if not isinstance(piece, bytes):
            raise TypeError(
                f"vocab[{index}] is {type(piece).__name__!r}, not bytes -- this compiler takes "
                "Sequence[bytes] (each token's own undecoded bytes), not Sequence[str]. Encode "
                "each vocabulary piece to bytes before calling (T-2908's byte-level port)."
            )
    active = _resolve_limits(limits)
    name_bytes = b"" if name is None else (name.encode("utf-8") if isinstance(name, str) else bytes(name))
    vocab = tuple(zero_special_ids(vocab, special_ids))

    # Exact schema properties first: nesting depth, then every array's minItems (in `_groups`).
    depth = _schema_depth(schema, active["recursion_depth"])
    arrays: dict[str, int] = {}
    groups = _groups(schema, "$", limits=active, arrays=arrays)

    admission = _Admission(active, len(vocab))
    char_dfa, char_start, char_accepting, content_states = _char_dfa(groups, admission)
    admission.trie()
    trie = _vocab_trie(vocab)
    admission.section_header(len(name_bytes))

    admission.token_state()
    ids: dict[int, int] = {char_start: 0}
    prefixes: dict[int, bytes] = {0: b""}
    transitions: dict[int, dict[int, int]] = {}
    accepting: set[int] = set()
    queue: deque[int] = deque([char_start])
    while queue:
        current = queue.popleft()
        state = ids[current]
        if current in char_accepting:
            admission.accepting_state()
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
        admission.transitions(len(targets))
        row: dict[int, int] = {}
        for token_id in sorted(targets):
            target = targets[token_id]
            if target not in ids:
                admission.token_state()
                ids[target] = len(ids)
                prefixes[ids[target]] = prefixes[state] + vocab[token_id]
                queue.append(target)
            row[token_id] = ids[target]
        transitions[state] = row

    dead = _states_that_cannot_finish(transitions, accepting)
    if dead:
        state = dead[0]
        char_state = next(c for c, s in ids.items() if s == state)
        completion = _shortest_completion(char_dfa, char_accepting, char_state)
        reason = (
            f"{len(dead)} reachable state(s) can never reach an accepting state: every token "
            f"sequence from them loops or stops short of a complete document (the byte completion "
            f"{completion!r} has no spelling in the vocabulary from there)"
        )
        raise SchemaCompileError(
            f"state {state} cannot reach an accepting state (dead end) after the serialization "
            f"prefix {prefixes[state]!r}: {reason}",
            state_id=state,
            prefix=prefixes[state],
            reason=reason,
        )

    counts = {"minItems": dict(arrays), "recursion_depth": depth, **admission.counts}
    return MaskPages(vocab, transitions, 0, frozenset(accepting), resource_counts=counts)


# --- SCM1 serialization (the attachment command's writer) -----------------------------------


def fnv1a64(data: bytes) -> int:
    """FNV-1a 64 (standard offset basis and prime), the SCM1 schema-name hash
    (`include/superslm/schema_masks.h::Fnv1a64`)."""
    h = 0xCBF29CE484222325
    for byte_val in data:
        h = ((h ^ byte_val) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def serialize_scm1(named_schemas: Sequence[tuple[str | bytes, MaskPages]], vocab_size: int) -> bytes:
    """The SCM1 section for `named_schemas`, in the byte format of design Sec13.2 as the T-2132
    fixture mechanism writes it (`tools/t2132_build_g5_fixture.py::_serialize_scm1`): a 24-byte
    header, one 56-byte descriptor per schema, the name blob, then per schema the mask pages,
    the CSR state-offset array, the ascending accepting ids and the (token, target) pairs in
    ascending token order. State ids must be dense `[0, N)` (MaskPages' own BFS numbering)."""
    import struct

    names = [n.encode("utf-8") if isinstance(n, str) else bytes(n) for n, _ in named_schemas]
    name_blob = b"".join(names)
    cursor = _SCM1_HEADER_BYTES + _SCM1_DESCRIPTOR_BYTES * len(named_schemas) + len(name_blob)
    page_bytes = (vocab_size + 7) // 8
    descriptors: list[bytes] = []
    blocks: list[bytes] = []
    name_off = 0
    for raw_name, (_, mp) in zip(names, named_schemas):
        states = sorted(set(mp.transitions) | {mp.start} |
                        {t for row in mp.transitions.values() for t in row.values()})
        state_count = max(states) + 1 if states else 1
        if states != list(range(state_count)):
            raise ValueError("MaskPages state ids are not dense [0, N)")
        masks_off = cursor
        for s in range(state_count):
            page = mp.page(s)
            if len(page) != page_bytes:
                raise ValueError(f"mask page width {len(page)} disagrees with vocab_size {vocab_size}")
            blocks.append(page)
        cursor += state_count * page_bytes
        offsets_off = cursor
        row_starts = [0]
        flat = bytearray()
        for s in range(state_count):
            row = mp.transitions.get(s, {})
            for token in sorted(row):
                flat += struct.pack("<II", token, row[token])
            row_starts.append(row_starts[-1] + len(row))
        blocks.append(struct.pack(f"<{len(row_starts)}I", *row_starts))
        cursor += len(row_starts) * 4
        accepting_off = cursor
        accepting_sorted = sorted(mp.accepting)
        blocks.append(struct.pack(f"<{len(accepting_sorted)}I", *accepting_sorted))
        cursor += len(accepting_sorted) * 4
        transitions_off = cursor
        blocks.append(bytes(flat))
        cursor += len(flat)
        descriptors.append(struct.pack(
            "<IIIIIIQQQQ", name_off, len(raw_name), state_count, len(accepting_sorted), row_starts[-1], 0,
            masks_off, offsets_off, accepting_off, transitions_off))
        name_off += len(raw_name)
    header = struct.pack("<4sIIIII", b"SCM1", 1, len(named_schemas), vocab_size, len(name_blob), 0)
    return header + b"".join(descriptors) + name_blob + b"".join(blocks)

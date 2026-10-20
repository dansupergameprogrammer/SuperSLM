"""Oracle B for the array converter (plan of record §7, §4.1, §6 row 7): the exact canonical byte and token
language of a schema in the converter's subset, specified from the published converter contract.

It imports no converter, NFA, MaskPages, token-target or accepting-set code. The rules it checks:
- an object serializes as `{`, its properties in `properties` order, each as `"key":value`, `,` between
  them, then `}`; every property is required and none is added;
- an enum value is exactly `json.dumps(value, ensure_ascii=False)`; a boolean is `true` or `false`;
- a free string leaf is `"`, content, `"`. Content bytes are strict UTF-8 (no overlong form, no encoded
  surrogate, nothing past U+10FFFF), no raw byte below 0x20, `"` and `\\` only escaped, escapes are the
  eight short forms or `\\u` with four hex digits of either case. The decoded value must hold at least one
  unit outside the insignificant set `{}[],:` plus space, tab, CR and LF;
- an array is `[`, at least `minItems` items separated by single commas, then `]`; `[]` only when
  `minItems` is 0; no trailing or leading comma;
- no byte outside this grammar anywhere (no free whitespace), and the document ends exactly at the end;
- token boundaries: the opening quote of every free-string value is a token by itself (a token may not
  carry structure into a value, nor open a value and continue into it); closing is unrestricted;
- token identity, when a vocabulary is given: a special id is never admitted, and of several ids that
  spell the same bytes only the lowest is admitted.
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

INSIGNIFICANT = frozenset("{}[],: \t\r\n")
SHORT_ESCAPES = {ord('"'): '"', ord("\\"): "\\", ord("/"): "/", ord("b"): "\b",
                 ord("f"): "\f", ord("n"): "\n", ord("r"): "\r", ord("t"): "\t"}
HEX = frozenset(b"0123456789abcdefABCDEF")


class Reject(ValueError):
    """Oracle B refuses the bytes or the tokenization; the message names the rule."""


def node_kind(node: dict) -> str:
    if "enum" in node:
        return "enum"
    kind = node.get("type")
    if kind in ("object", "array", "string", "boolean"):
        return kind
    raise Reject(f"schema node outside the subset: {node!r}")


# ------------------------------------------------------------------ byte-level grammar

def _string(raw: bytes, pos: int) -> tuple[int, str]:
    """Parse one free-string value starting at its opening quote. Returns (end, decoded)."""
    if pos >= len(raw) or raw[pos] != 0x22:
        raise Reject(f"expected string opening quote at {pos}")
    i = pos + 1
    units: list[str] = []
    run = bytearray()

    def flush():
        if run:
            try:
                units.append(bytes(run).decode("utf-8", errors="strict"))
            except UnicodeDecodeError as error:
                raise Reject(f"invalid UTF-8 in string content near {pos}: {error}") from None
            run.clear()

    while True:
        if i >= len(raw):
            raise Reject(f"unterminated string opened at {pos}")
        b = raw[i]
        if b == 0x22:
            flush()
            i += 1
            break
        if b < 0x20:
            raise Reject(f"raw control byte 0x{b:02x} in string at {i}")
        if b == 0x5C:
            flush()
            if i + 1 >= len(raw):
                raise Reject(f"truncated escape at {i}")
            e = raw[i + 1]
            if e in SHORT_ESCAPES:
                units.append(SHORT_ESCAPES[e])
                i += 2
                continue
            if e == ord("u"):
                digits = raw[i + 2:i + 6]
                if len(digits) != 4 or any(d not in HEX for d in digits):
                    raise Reject(f"bad \\u escape at {i}")
                units.append(chr(int(digits, 16)))
                i += 6
                continue
            raise Reject(f"bad escape \\{chr(e)!r} at {i}")
        run.append(b)
        i += 1
    decoded = "".join(units)
    if not any(ch not in INSIGNIFICANT for ch in decoded):
        raise Reject(f"empty or insignificant string value at {pos}")
    return i, decoded


def _parse(node: dict, raw: bytes, pos: int, path: str, openings: list[int]) -> tuple[int, Any]:
    kind = node_kind(node)
    if kind == "enum":
        matches = []
        for value in node["enum"]:
            spelled = json.dumps(value, ensure_ascii=False).encode("utf-8")
            if raw.startswith(spelled, pos):
                matches.append((pos + len(spelled), value))
        if not matches:
            raise Reject(f"no enum value at {path} (offset {pos})")
        # Canonical string enums are prefix-free once quoted; take the longest for other literals.
        return max(matches, key=lambda m: m[0])
    if kind == "boolean":
        for spelled, value in ((b"true", True), (b"false", False)):
            if raw.startswith(spelled, pos):
                return pos + len(spelled), value
        raise Reject(f"no boolean at {path} (offset {pos})")
    if kind == "string":
        openings.append(pos)
        return _string(raw, pos)
    if kind == "object":
        if not raw.startswith(b"{", pos):
            raise Reject(f"expected '{{' at {path} (offset {pos})")
        pos += 1
        out = {}
        for index, (key, child) in enumerate(node["properties"].items()):
            if index:
                if not raw.startswith(b",", pos):
                    raise Reject(f"expected ',' before {path}.{key} (offset {pos})")
                pos += 1
            spelled = (json.dumps(key, ensure_ascii=False) + ":").encode("utf-8")
            if not raw.startswith(spelled, pos):
                raise Reject(f"expected key {key!r} at {path} (offset {pos})")
            pos += len(spelled)
            pos, out[key] = _parse(child, raw, pos, f"{path}.{key}", openings)
        if not raw.startswith(b"}", pos):
            raise Reject(f"expected '}}' closing {path} (offset {pos})")
        return pos + 1, out
    # array
    minimum = node.get("minItems", 0)
    if not raw.startswith(b"[", pos):
        raise Reject(f"expected '[' at {path} (offset {pos})")
    pos += 1
    items = []
    if raw.startswith(b"]", pos):
        pos += 1
    else:
        while True:
            pos, item = _parse(node["items"], raw, pos, f"{path}[{len(items)}]", openings)
            items.append(item)
            if raw.startswith(b",", pos):
                pos += 1
                continue
            if raw.startswith(b"]", pos):
                pos += 1
                break
            raise Reject(f"expected ',' or ']' in {path} (offset {pos})")
    if len(items) < minimum:
        raise Reject(f"{path} has {len(items)} items, minimum {minimum}")
    return pos, items


def parse(schema: dict, raw: bytes) -> tuple[Any, list[int]]:
    """Byte-level verdict. Returns (value, string-value opening offsets) or raises Reject."""
    openings: list[int] = []
    end, value = _parse(schema, raw, 0, "$", openings)
    if end != len(raw):
        raise Reject(f"bytes after the complete document at offset {end}")
    return value, openings


# ------------------------------------------------------------------ token level

def check_pieces(schema: dict, raw: bytes, pieces: Sequence[bytes]) -> Any:
    """Token-level verdict over supplied token byte pieces."""
    if any(not p for p in pieces):
        raise Reject("empty token piece")
    if b"".join(pieces) != raw:
        raise Reject("token pieces do not spell the output")
    value, openings = parse(schema, raw)
    spans = {}
    cursor = 0
    for p in pieces:
        spans[cursor] = cursor + len(p)
        cursor += len(p)
    for opening in openings:
        if spans.get(opening) != opening + 1:
            raise Reject(f"compound opening: the string value opened at byte {opening} is not a one-byte quote token")
    return value


class Vocabulary:
    """Token bytes by id, with the special ids and the lowest id of every spelling."""

    def __init__(self, pieces: Sequence[bytes], special_ids: Iterable[int]):
        self.pieces = tuple(pieces)
        self.special = frozenset(special_ids)
        first: dict[bytes, int] = {}
        for tid, piece in enumerate(self.pieces):
            if piece and tid not in self.special and piece not in first:
                first[piece] = tid
        self.first = first

    def id_of(self, piece: bytes) -> int:
        return self.first[piece]

    def ids(self, pieces: Iterable[bytes]) -> list[int]:
        return [self.first[p] for p in pieces]

    def bytewise(self, raw: bytes) -> list[int]:
        return [self.first[bytes([b])] for b in raw]


def check_tokens(schema: dict, token_ids: Sequence[int], vocab: Vocabulary) -> Any:
    for tid in token_ids:
        if tid in vocab.special:
            raise Reject(f"special id {tid} admitted")
        piece = vocab.pieces[tid]
        if not piece:
            raise Reject(f"id {tid} has no bytes")
        if vocab.first.get(piece) != tid:
            raise Reject(f"id {tid} is not the lowest id spelling {piece!r}")
    pieces = [vocab.pieces[t] for t in token_ids]
    return check_pieces(schema, b"".join(pieces), pieces)


def accepts(schema: dict, raw: bytes) -> bool:
    try:
        parse(schema, raw)
        return True
    except Reject:
        return False


# ------------------------------------------------------------------ canonical serializer

def serialize(schema: dict, value: Any) -> bytes:
    """Canonical bytes of `value` under `schema` (compact, properties order, ensure_ascii=False)."""
    def ser(node, v):
        kind = node_kind(node)
        if kind == "object":
            if list(v) != list(node["properties"]):
                raise Reject("value keys differ from schema properties")
            return "{" + ",".join(json.dumps(k, ensure_ascii=False) + ":" + ser(node["properties"][k], v[k])
                                  for k in node["properties"]) + "}"
        if kind == "array":
            return "[" + ",".join(ser(node["items"], x) for x in v) + "]"
        return json.dumps(v, ensure_ascii=False)
    return ser(schema, value).encode("utf-8")

"""T-3094 independent format, language, consumer, and grading references.

This file imports no converter, NFA, MaskPages, tokenizer-target or accepting-set code.
Its results remain QUARANTINED until an independent seat commissions the instruments.
"""
from __future__ import annotations

import json
import re
import struct

FIELDS = ("action", "subject", "target", "trigger", "depends_on_previous")
ACTIONS = frozenset(("fire", "move", "hold", "wait", "scan"))
INSIGNIFICANT = frozenset("{}[],: \t\r\n")


def _strict_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate key: {key}")
        out[key] = value
    return out


def parse_shape(raw: bytes, minimum: int = 1) -> dict:
    """Oracle A: parsed semantic shape; accepts key order and whitespace variants."""
    document = json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_object)
    if type(document) is not dict or set(document) != {"commands"}:
        raise ValueError("root shape")
    items = document["commands"]
    if type(items) is not list or len(items) < minimum:
        raise ValueError("commands shape/minimum")
    for item in items:
        if type(item) is not dict or set(item) != set(FIELDS):
            raise ValueError("command fields")
        if type(item["action"]) is not str or item["action"] not in ACTIONS:
            raise ValueError("action enum")
        for field in ("subject", "target", "trigger"):
            if type(item[field]) is not str:
                raise ValueError(f"{field} type")
        if type(item["depends_on_previous"]) is not bool:
            raise ValueError("dependency type")
    return document


class _Canonical:
    """Grammar walk over decoded text, independent of compiler state construction."""

    def __init__(self, raw: bytes):
        self.text = raw.decode("utf-8")
        self.position = 0
        self.openings: list[int] = []
        self.decoder = json.JSONDecoder()

    def literal(self, value: str):
        if not self.text.startswith(value, self.position):
            raise ValueError(f"expected {value!r} at {self.position}")
        self.position += len(value)

    def string(self, meaningful: bool = True, content_leaf: bool = True) -> str:
        if self.position >= len(self.text) or self.text[self.position] != '"':
            raise ValueError("expected string")
        if content_leaf:
            self.openings.append(len(self.text[:self.position].encode("utf-8")))
        value, end = self.decoder.raw_decode(self.text, self.position)
        if type(value) is not str:
            raise ValueError("expected string value")
        if meaningful and not any(character not in INSIGNIFICANT for character in value):
            raise ValueError("insignificant string")
        self.position = end
        return value

    def command(self):
        self.literal("{")
        for index, field in enumerate(FIELDS):
            if index:
                self.literal(",")
            self.literal(json.dumps(field) + ":")
            if field == "action":
                action_start = self.position
                action = self.string(content_leaf=False)
                if action not in ACTIONS or self.text[action_start:self.position] != json.dumps(action):
                    raise ValueError("action enum")
            elif field == "depends_on_previous":
                if self.text.startswith("true", self.position):
                    self.literal("true")
                else:
                    self.literal("false")
            else:
                self.string()
        self.literal("}")


def canonical_walk(raw: bytes, pieces: tuple[bytes, ...], minimum: int = 1) -> dict:
    """Oracle B: exact syntax and token boundaries over actual supplied token pieces."""
    if b"".join(pieces) != raw or any(not piece for piece in pieces):
        raise ValueError("token pieces do not form output")
    parser = _Canonical(raw)
    parser.literal('{"commands":[')
    count = 0
    if not parser.text.startswith("]", parser.position):
        parser.command()
        count = 1
        while parser.text.startswith(",", parser.position):
            parser.literal(",")
            parser.command()
            count += 1
    parser.literal("]}")
    if parser.position != len(parser.text) or count < minimum:
        raise ValueError("incomplete document or array minimum")
    starts = []
    cursor = 0
    for piece in pieces:
        starts.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    for opening in parser.openings:
        # Value opening is permitted only as a one-byte quote token. A token
        # containing preceding structure or following content is refused.
        if not any(start == opening and end == opening + 1 for start, end in starts):
            raise ValueError("compound opening crosses string boundary")
    return parse_shape(raw, minimum)


def scm1_account(state_count: int, transition_count: int, accepting_count: int,
                 vocab_size: int, name_bytes: int = 1) -> dict:
    """Independent wire-size arithmetic for one schema, before section framing."""
    page_bytes = (vocab_size + 7) // 8
    blocks = {
        "mask_pages": state_count * page_bytes,
        "state_offsets": (state_count + 1) * 4,
        "accepting": accepting_count * 4,
        "transitions": transition_count * 8,
    }
    return {**blocks, "total": 24 + 56 + name_bytes + sum(blocks.values())}


def parse_scm1(data: bytes) -> dict:
    if len(data) < 80:
        raise ValueError("short SCM1")
    magic, version, count, vocab, name_len, reserved = struct.unpack_from("<4sIIIII", data)
    if magic != b"SCM1" or version != 1 or count != 1 or reserved != 0:
        raise ValueError("SCM1 header")
    noff, nlen, states, accepts, transitions, dreserved, masks_at, offsets_at, accepts_at, trans_at = struct.unpack_from(
        "<IIIIIIQQQQ", data, 24)
    if dreserved != 0 or noff + nlen > name_len:
        raise ValueError("SCM1 descriptor")
    block = scm1_account(states, transitions, accepts, vocab, name_len)
    if (masks_at, offsets_at, accepts_at, trans_at) != (
        80 + name_len,
        80 + name_len + block["mask_pages"],
        80 + name_len + block["mask_pages"] + block["state_offsets"],
        80 + name_len + block["mask_pages"] + block["state_offsets"] + block["accepting"],
    ) or len(data) != block["total"]:
        raise ValueError("SCM1 block offsets/length")
    offsets = struct.unpack_from(f"<{states + 1}I", data, offsets_at)
    if offsets[0] != 0 or offsets[-1] != transitions or tuple(sorted(offsets)) != offsets:
        raise ValueError("CSR offsets")
    accepting = struct.unpack_from(f"<{accepts}I", data, accepts_at)
    if len(set(accepting)) != accepts or any(state >= states for state in accepting):
        raise ValueError("accepting states")
    for state in range(states):
        page = data[masks_at + state * ((vocab + 7) // 8): masks_at + (state + 1) * ((vocab + 7) // 8)]
        seen = set()
        for entry in range(offsets[state], offsets[state + 1]):
            token, target = struct.unpack_from("<II", data, trans_at + 8 * entry)
            if token >= vocab or target >= states or token in seen or not (page[token // 8] & (1 << (token % 8))):
                raise ValueError("CSR transition/mask disagreement")
            seen.add(token)
        if any(bool(page[token // 8] & (1 << (token % 8))) != (token in seen) for token in range(vocab)):
            raise ValueError("mask has transition mismatch")
    return {"name": data[80 + noff:80 + noff + nlen].decode("utf-8"), "states": states,
            "transitions": transitions, "accepting": tuple(accepting), "vocab": vocab}


def write_scm1(name: str, vocab_size: int, rows: dict[int, dict[int, int]],
               accepting: frozenset[int]) -> bytes:
    """Independent SCM1 wire oracle from abstract token rows, one named schema."""
    name_raw = name.encode("utf-8")
    states = max(rows.keys() | accepting | {0}) + 1
    if set(rows) != set(range(states)):
        raise ValueError("noncontiguous state ids")
    page_width = (vocab_size + 7) // 8
    pages = bytearray(states * page_width)
    offsets = [0]
    transitions = bytearray()
    for state in range(states):
        for token, target in sorted(rows[state].items()):
            if not 0 <= token < vocab_size or not 0 <= target < states:
                raise ValueError("invalid transition")
            pages[state * page_width + token // 8] |= 1 << (token % 8)
            transitions.extend(struct.pack("<II", token, target))
        offsets.append(len(transitions) // 8)
    offsets_raw = struct.pack(f"<{len(offsets)}I", *offsets)
    accepting_raw = struct.pack(f"<{len(accepting)}I", *sorted(accepting))
    start = 80 + len(name_raw)
    descriptor = struct.pack("<IIIIIIQQQQ", 0, len(name_raw), states, len(accepting),
                             offsets[-1], 0, start, start + len(pages),
                             start + len(pages) + len(offsets_raw),
                             start + len(pages) + len(offsets_raw) + len(accepting_raw))
    header = struct.pack("<4sIIIII", b"SCM1", 1, 1, vocab_size, len(name_raw), 0)
    return header + descriptor + name_raw + pages + offsets_raw + accepting_raw + transitions


def consumer_contract(raw: bytes | None, *, route: str, stop: str, accepting: bool,
                      cap: int, domain_valid: bool = True) -> tuple[str, tuple[dict, ...], int, int]:
    """Reference status, exposed batch, query starts, action attempts."""
    if route in ("NoAction", "Cancelled"):
        return route, (), 0, 0
    if route != "Executable" or cap <= 0:
        raise ValueError("invalid router outcome/cap")
    if stop != "Complete" or not accepting:
        status = "BudgetStop" if stop == "Budget" else ("SchemaDeadEnd" if stop == "DeadEnd" else "NonacceptingStop")
        return status, (), 1, 0
    try:
        document = parse_shape(raw or b"")
    except (ValueError, UnicodeError, json.JSONDecodeError):
        return "InvalidDocument", (), 1, 0
    items = document["commands"]
    if len(items) > cap:
        return "OverCap", (), 1, 0
    if not domain_valid:
        return "DomainRejected", (), 1, 0
    return "Accepted", tuple(items), 1, 0


def grade_pc2(pairs: list[tuple[tuple[dict, ...], tuple[dict, ...]]],
              predictions: list[tuple[tuple[dict, ...], tuple[dict, ...]]]) -> dict:
    """Whole pair and arm counts, not a pooled accuracy that hides policy errors."""
    if len(pairs) != 40 or len(predictions) != 40:
        raise ValueError("each condition requires 40 unique pairs")
    left = sum(pred[0] == gold[0] for gold, pred in zip(pairs, predictions))
    right = sum(pred[1] == gold[1] for gold, pred in zip(pairs, predictions))
    both = sum(pred == gold for gold, pred in zip(pairs, predictions))
    return {"trigger_arm": left, "tracked_wait_arm": right, "pairs": both,
            "passes": min(left, right, both) >= 38}


def grade_semantics(gold: list[dict], predicted: list[dict]) -> dict:
    """Frozen exact metrics over adjudicated rows; model output never sets bars."""
    if len(gold) != len(predicted) or not gold:
        raise ValueError("gold/prediction population mismatch")
    if len({r["id"] for r in gold}) != len(gold) or [r["id"] for r in gold] != [r["id"] for r in predicted]:
        raise ValueError("missing, duplicated or reordered IDs")
    executable = [(g, p) for g, p in zip(gold, predicted) if g["status"] == "Executable"]
    if not executable:
        raise ValueError("no executable denominator")
    count_ok = sum(p.get("status") == "Executable" and
                   len(g["commands"]) == len(p.get("commands", [])) for g, p in executable)
    ordered_ok = sum(p.get("status") == "Executable" and
                     [c["action"] for c in g["commands"]] ==
                     [c.get("action") for c in p.get("commands", [])] for g, p in executable)
    whole_ok = sum(p.get("status") == "Executable" and
                   g["commands"] == p.get("commands") for g, p in executable)
    field = {}
    for name in FIELDS:
        denominator = sum(len(g["commands"]) for g, _ in executable)
        numerator = sum(
            p.get("status") == "Executable" and idx < len(p.get("commands", []))
            and item[name] == p["commands"][idx].get(name)
            for g, p in executable for idx, item in enumerate(g["commands"]))
        field[name] = (numerator, denominator)
    no_action = {}
    for status in ("Cancelled", "NoAction"):
        subset = [(g, p) for g, p in zip(gold, predicted) if g["status"] == status]
        no_action[status] = {"correct": sum(p.get("status") == status and p.get("query_starts") == 0
                                            and p.get("action_attempts") == 0
                                            and not p.get("commands") and p.get("exposed_count", 0) == 0
                                            for _, p in subset),
                             "total": len(subset)}
    strata = {}
    for g, p in executable:
        name = g.get("source_class", "fixture")
        part = strata.setdefault(name, [0, 0])
        part[1] += 1
        part[0] += int(p.get("status") == "Executable" and g["commands"] == p.get("commands"))
    return {"count": (count_ok, len(executable)), "ordered_action": (ordered_ok, len(executable)),
            "whole_document": (whole_ok, len(executable)), "fields": field, "no_action": no_action,
            "strata": {name: tuple(value) for name, value in strata.items()}}


def semantic_pass(report: dict) -> bool:
    """Absolute bars; caller separately checks each stratum and PC-2 condition."""
    rate = lambda part: part[0] / part[1] if part[1] else 0.0
    return (rate(report["count"]) >= 0.95 and rate(report["ordered_action"]) >= 0.95
            and rate(report["whole_document"]) >= 0.85
            and all(rate(v) >= 0.95 for v in report["fields"].values())
            and all(rate(v) >= 0.80 for v in report["strata"].values())
            and all(v["correct"] == v["total"] and v["total"] >= 100
                    for v in report["no_action"].values()))

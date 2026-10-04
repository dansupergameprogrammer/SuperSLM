"""T-3103 S0 reference for `ResolvedCondition` canonical form and the S3 final-guard check
(plan rev 15 s4.2, D-SLM8082). Written from the plan text; imports no product code.

A resolved command is {"action", "subject", "target", "trigger", "after"}. `trigger` holds the four
groups start_all, start_any, stop_any and during_all. Each entry is {"instance", "predicate", "kind",
"cmp", "value", "negated", "binding": {"mode", "ref"}}. The resolver owns the canonical order. The S3 guard
re-checks it, and also checks that every group is sorted with no duplicate entries, every rational is in
lowest terms and in range, and every wait ref names an earlier ordinal of an inserted wait. A violation
refuses as InvalidResolvedCommand.
"""
from __future__ import annotations

import math

GROUPS = ("start_all", "start_any", "stop_any", "during_all")
KINDS = ("state", "event", "numeric", "count")
MODES = ("passive", "reuse", "new", "wait")
CMPS = ("none", "below", "above", "at_most", "at_least", "reaches")
LIMIT = 2 ** 31 - 1


class GuardRefusal(ValueError):
    status = "InvalidResolvedCommand"


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _ref_key(ref):
    if ref is None:
        return (0, 0, "")
    if _is_int(ref):
        return (1, ref, "")
    return (2, 0, ref)


class _Entry:
    """Total order of s4.2: (instance, predicate, kind, cmp, value, negated, binding.mode, binding.ref)."""

    __slots__ = ("e",)

    def __init__(self, e):
        self.e = e

    def _tuple(self):
        e = self.e
        return (e["instance"], e["predicate"], e["kind"], e["cmp"])

    def __lt__(self, other):
        a, b = self.e, other.e
        if self._tuple() != other._tuple():
            return self._tuple() < other._tuple()
        va, vb = a["value"], b["value"]
        if (va is None) != (vb is None):
            return va is None
        if va is not None and va["num"] * vb["den"] != vb["num"] * va["den"]:
            return va["num"] * vb["den"] < vb["num"] * va["den"]
        if a["negated"] != b["negated"]:
            return a["negated"] is False
        if a["binding"]["mode"] != b["binding"]["mode"]:
            return a["binding"]["mode"] < b["binding"]["mode"]
        return _ref_key(a["binding"]["ref"]) < _ref_key(b["binding"]["ref"])

    def __eq__(self, other):
        return not (self < other) and not (other < self)


def canonicalize(trigger: dict) -> dict:
    """Resolver-side canonical form: every group sorted, exact duplicates collapsed."""
    out = {}
    for g in GROUPS:
        entries = sorted(trigger.get(g, []), key=_Entry)
        dedup = []
        for e in entries:
            if not dedup or e != dedup[-1]:
                dedup.append(e)
        out[g] = dedup
    return out


def check_entry(e, ordinal, wait_ordinals):
    if set(e) != {"instance", "predicate", "kind", "cmp", "value", "negated", "binding"}:
        raise GuardRefusal("entry fields")
    if not isinstance(e["instance"], str) or not isinstance(e["predicate"], str) or e["kind"] not in KINDS \
            or e["cmp"] not in CMPS or not isinstance(e["negated"], bool):
        raise GuardRefusal("entry types")
    v = e["value"]
    if v is not None:
        if not isinstance(v, dict) or set(v) != {"num", "den"} or not _is_int(v["num"]) or not _is_int(v["den"]):
            raise GuardRefusal("value is not an integer rational")
        if v["den"] < 1 or abs(v["num"]) > LIMIT or v["den"] > LIMIT or math.gcd(abs(v["num"]), v["den"]) != 1:
            raise GuardRefusal("rational not in lowest terms or range")
    if (e["cmp"] == "none") != (v is None):
        raise GuardRefusal("cmp/value mismatch")
    b = e["binding"]
    if not isinstance(b, dict) or set(b) != {"mode", "ref"} or b["mode"] not in MODES:
        raise GuardRefusal("binding")
    if b["mode"] == "wait":
        if not _is_int(b["ref"]) or b["ref"] >= ordinal or b["ref"] not in wait_ordinals:
            raise GuardRefusal("wait ref is not an earlier inserted wait")
    elif b["mode"] == "reuse":
        if not isinstance(b["ref"], str):
            raise GuardRefusal("reuse ref")
    elif b["ref"] is not None:
        raise GuardRefusal("ref on passive/new binding")


def guard_batch(commands: list[dict], wait_ordinals: set[int] = frozenset()) -> None:
    """Raises GuardRefusal (status InvalidResolvedCommand) on any non-canonical command."""
    for i, c in enumerate(commands):
        if set(c) != {"action", "subject", "target", "trigger", "after"}:
            raise GuardRefusal("command fields")
        t = c["trigger"]
        if not isinstance(t, dict) or set(t) != set(GROUPS):
            raise GuardRefusal("trigger groups")
        after = c["after"]
        if not isinstance(after, list) or any(not _is_int(a) or a < 0 or a >= i for a in after) \
                or after != sorted(set(after)):
            raise GuardRefusal("after list")
        for g in GROUPS:
            entries = t[g]
            for e in entries:
                check_entry(e, i, wait_ordinals)
            for a, b in zip(entries, entries[1:]):
                if not (_Entry(a) < _Entry(b)):
                    raise GuardRefusal(f"{g} not sorted or duplicate")

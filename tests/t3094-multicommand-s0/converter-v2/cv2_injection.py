"""Injection pairs (plan §4.1: "the test harness may inject smaller numeric limits through the same
production guard to isolate each under/over pair without an excessive run"; D-SLM8120 for R_MEM).

Every pair is built from IntentDoc v2 itself, so the array path is on the guarded path, and is counted by
count model v2 at the byte vocabulary, never by the converter. The injected limit replaces only the target;
every other limit stays at its release value. For the exact schema properties (`minItems`, depth) the
injected limit is the under value and the over schema is the next value. For the five size counts the
injected limit is the geometric mean of the model's under and over counts, with at least a factor 1.5 on
each side, so the verdict does not depend on the converter's counter equalling the model's to the unit
(§4.1: "S1's front end may build a different NFA"; the tracked-byte model is a model, not a measurement).
"""
from __future__ import annotations

import math

import count_model_v2 as cm
import cv2_fixtures as F


def _extend_enum(location, n, width):
    """`n` extra enum values with distinct first bytes after a two-digit index, so each adds about `width`
    byte-chain states that no other value shares."""
    s = F.fresh()
    leaf = F.node(s, location)
    letters = "abcdefghijklmnopqrstuvwxyz"
    leaf["enum"] = list(leaf["enum"]) + [
        f"{i:02d}" + "".join(letters[(i * 7 + j * 3) % 26] for j in range(width)) for i in range(n)]
    return s


def _extra_strings(location, n):
    s = F.fresh()
    props = F.node(s, location)["properties"]
    for i in range(n):
        props[f"note{i}"] = {"type": "string"}
    F.node(s, location)["required"] = list(props)
    return s


PAIRS = {
    "minItems_root": (lambda: F.with_min(1, 0), lambda: F.with_min(2, 0), "minItems", "exact"),
    "minItems_inner": (lambda: F.with_min(1, 1), lambda: F.with_min(1, 2), "minItems", "exact"),
    "recursion_depth": (lambda: F.fresh(), lambda: F.deepen_value(1), "recursion_depth", "exact"),
    "nfa_states": (lambda: F.fresh(), lambda: _extend_enum("action", 60, 30), "nfa_states", "mean"),
    "dfa_states": (lambda: F.fresh(), lambda: _extend_enum("watch", 60, 18), "dfa_states", "mean"),
    "token_transitions": (lambda: F.fresh(), lambda: _extra_strings("intent", 14), "token_transitions", "mean"),
    "serialized_mask_bytes": (lambda: F.fresh(), lambda: _extra_strings("root", 14), "serialized_mask_bytes", "mean"),
    "tracked_temporary_bytes": (lambda: F.fresh(), lambda: _extra_strings("atom", 16), "tracked_temporary_bytes", "mean"),
}


def counts(schema):
    c = cm.count(schema, "intents")
    out = F.model_counts_to_contract(c)
    out["minItems"] = max(c["minItems"].values(), default=0)
    return out


def limits(key, value):
    lim = dict(F.RELEASE)
    lim[key] = value
    return lim


def crossing(c, lim):
    return [k for k in F.LIMIT_KEYS if c[k] > lim[k]]


def pair(name):
    make_under, make_over, key, rule = PAIRS[name]
    under, over = make_under(), make_over()
    if rule == "exact":
        limit = counts(under)[key]
    else:
        limit = int(math.sqrt(counts(under)[key] * counts(over)[key]))
    return under, over, key, limit


# ------------------------------------------------------------------ the same pairs at the pinned vocabulary
#
# The attach command compiles at the checkpoint's (pinned) vocabulary, so its injected limits are derived there
# (S1 builder finding: the byte-vocabulary limits for transitions, section and tracked bytes sit below the
# under schema's pinned counts, so no correct converter could admit it). minItems and depth do not depend on the
# vocabulary and keep their byte pairs. At the pinned vocabulary the byte over schemas for section and tracked
# bytes also cross the release transition limit, so those two get over schemas that grow mask pages
# (enum values: states with few transitions) instead; test_cv2_reference checks every pair's isolation.

def _with_enum(schema, location, n, width=40):
    letters = "abcdefghijklmnopqrstuvwxyz"
    leaf = F.node(schema, location)
    leaf["enum"] = list(leaf["enum"]) + [
        f"{i:02d}" + "".join(letters[(i * 7 + j * 3) % 26] for j in range(width)) for i in range(n)]
    return schema


PINNED_OVER = {
    "serialized_mask_bytes": lambda: _with_enum(F.fresh(), "watch", 33),
    # Re-derived after S1 charged the trie (79.4 MB at the pinned vocabulary, in both sides): 12 intent strings
    # and 28 extra `watch` values give over/under 1.99 with transitions, section and states admitted.
    "tracked_temporary_bytes": lambda: _with_enum(_extra_strings("intent", 12), "watch", 28),
}
_PINNED_COUNTS = {}


def counts_pinned(schema, pieces):
    import json
    key = json.dumps(schema)
    if key not in _PINNED_COUNTS:
        c = cm.count(schema, "intents", pieces, F.REAL_VOCAB_SIZE)
        out = F.model_counts_to_contract(c, "pinned")
        out["minItems"] = max(c["minItems"].values(), default=0)
        _PINNED_COUNTS[key] = out
    return _PINNED_COUNTS[key]


def pair_pinned(name, pieces):
    make_under, make_over, key, rule = PAIRS[name]
    if rule == "exact":
        return pair(name)
    under = make_under()
    over = PINNED_OVER.get(key, make_over)()
    limit = int(math.sqrt(counts_pinned(under, pieces)[key] * counts_pinned(over, pieces)[key]))
    return under, over, key, limit

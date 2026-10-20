"""Release-limit pairs on schemas that contain an array (S1 ruling R1: release limits bind only array schemas,
brief-s1-fix-round1-rulings.md item 1). The frozen S0 pairs R_DEPTH, R_NFA, R_TRANS and R_SECTION
(v2/resource-pairs-v2.json) have no array, so under R1 their over schemas are admitted; these replace them for the
release-limit cells. Each is the frozen family with one array property added (`z`: at least one "x"), re-sized
at the pinned vocabulary so that, under count model v2 with the S1 release limits (F.RELEASE) and the trie charge,
the under schema crosses nothing and the over schema crosses only its target. test_cv2_reference recounts every
pair and checks that isolation; the sizes below are what that search found (2026-10-20)."""
from __future__ import annotations

import count_model_v2 as cm
import cv2_fixtures as F
import resource_pairs_v2 as RP

ARRAY = {"type": "array", "minItems": 1, "items": {"type": "string", "enum": ["x"]}}
SECTION_NAME = "pair"


def with_array(schema):
    out = dict(schema)
    out["properties"] = dict(schema["properties"], z=ARRAY)
    out["required"] = list(out["properties"])
    return out


def nfa(pad):
    return with_array(RP.nfa_family(100, 30, pad))


def leaves(n, pad=0):
    return with_array(RP.leaves(n, pad))


# Sizes found by the search (see the module docstring); the reference cell recounts them.
NFA_PAD = 678
TRANS_LEAVES = 21
SECTION_PAD = 2867

PAIRS = {
    "R_DEPTH": (lambda: F.triple_array(), lambda: F.triple_array(True), "recursion_depth"),
    "R_NFA": (lambda: nfa(NFA_PAD), lambda: nfa(NFA_PAD + 1), "nfa_states"),
    "R_TRANS": (lambda: leaves(TRANS_LEAVES), lambda: leaves(TRANS_LEAVES + 1), "token_transitions"),
    "R_SECTION": (lambda: leaves(TRANS_LEAVES, SECTION_PAD), lambda: leaves(TRANS_LEAVES, SECTION_PAD + 1),
                  "serialized_mask_bytes"),
}
_COUNTS = {}


def counts(schema, pieces):
    import json
    key = json.dumps(schema)
    if key not in _COUNTS:
        c = cm.count(schema, SECTION_NAME, pieces, F.REAL_VOCAB_SIZE)
        out = F.model_counts_to_contract(c, "pinned")
        out["minItems"] = max(c["minItems"].values(), default=0)
        out["accepting"] = c["accepting"]
        out["minItems_map"] = c["minItems"]
        _COUNTS[key] = out
    return _COUNTS[key]


def pair(name, pieces):
    """The pair in F.pair's shape: under/over schemas and count model v2's counts of each (pinned vocabulary)."""
    make_under, make_over, key = PAIRS[name]
    under, over = make_under(), make_over()
    cu = counts(under, pieces)
    model = {"dfa_states": cu["dfa_states"], "token_transitions": cu["token_transitions"],
             "accepting": cu["accepting"], "recursion_depth": cu["recursion_depth"], "minItems": cu["minItems_map"],
             "section_bytes": cu["serialized_mask_bytes"]}
    return {"pair": name, "under_schema": under, "over_schema": over, "under": model, "target": key}

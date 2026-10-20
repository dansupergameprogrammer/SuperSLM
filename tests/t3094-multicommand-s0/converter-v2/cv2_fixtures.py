"""Shared fixtures and the S0 converter contract for the converter-v2 red suite (plan of record §4.1, §6, §7).

Nothing here imports converter code. The product contract the cells assume, stated once:
- compile: `compile_schema_to_mask_pages(schema, vocab, *, special_ids, limits=None)`. `limits` maps a
  subset of LIMIT_KEYS to positive integers and replaces only those release limits (§4.1: "the test
  harness may inject smaller numeric limits through the same production guard"). Admission is
  count <= limit; a count above its limit raises `SchemaCompileError` whose message contains that
  limit's DIAGNOSTICS string (the T-3094 strings, kept so one converter serves both suites).
- name: `compile_schema_to_mask_pages(..., name=NAME)` gives the section name, which the section-byte count
  includes (S0a fold 4); the cells that reconcile section bytes pass it.
- release limits bind only a schema that contains an array; `limits=` keys bind every schema (S1 ruling R1).
- counts: the returned `MaskPages` carries `resource_counts`, a mapping with every LIMIT_KEYS key; its
  `minItems` value maps each array's location to its minimum (§4.1 must-admit: "its actual minItems at
  each array, depth, NFA states, DFA states, transitions, section bytes and tracked bytes are recorded").
- attach: `tools/sslm_attach_schema.py --base B --ckpt C --schema NAME:PATH [--schema ...] --out O
  [--limits-json J]` (the T-3094 attachment contract, re-pointed to IntentDoc v2).
"""
from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
S0 = HERE.parent
V2 = S0 / "v2"
ROOT = S0.parents[1]
TOOLS = ROOT / "tools"

INTENTDOC = json.loads((V2 / "intentdoc-v2.schema.json").read_text(encoding="utf-8"))
G1 = json.loads((V2 / "g1.schema.json").read_text(encoding="utf-8"))
LIMITS_FREEZE = json.loads((V2 / "limits-freeze.json").read_text(encoding="utf-8"))
RESOURCE_PAIRS = json.loads((V2 / "resource-pairs-v2.json").read_text(encoding="utf-8"))

TOKENIZER_DIR = Path(os.environ.get("T3094_TOKENIZER_DIR", "/mnt/project-files/models/qwen2.5-0.5b-instruct-7ae5576"))
REAL_VOCAB_SIZE = 151936
BYTE_VOCAB = tuple(bytes([i]) for i in range(256))

# Injection/count keys (T-3094 names) and the frozen limits-freeze.json key each one reads.
FREEZE_KEY = {"minItems": "minItems", "recursion_depth": "recursion_depth", "nfa_states": "nfa_states",
              "dfa_states": "dfa_states", "token_transitions": "token_transitions",
              "serialized_mask_bytes": "section_bytes", "tracked_temporary_bytes": "tracked_bytes"}
LIMIT_KEYS = tuple(FREEZE_KEY)
DIAGNOSTICS = {
    "minItems": "minItems limit",
    "recursion_depth": "recursion depth limit",
    "nfa_states": "NFA state limit",
    "dfa_states": "DFA state limit",
    "token_transitions": "token transition limit",
    "serialized_mask_bytes": "serialized mask byte limit",
    "tracked_temporary_bytes": "tracked temporary memory limit",
}
RELEASE_S0 = {key: LIMITS_FREEZE["limits"][FREEZE_KEY[key]] for key in LIMIT_KEYS}
# S1 finalized the tracked-bytes limit from measured counts by the D-SLM8087 rule (plan §4.1; conductor ruling,
# brief-s1-fix-round1-rulings.md item 3): the converter now charges the vocabulary trie it builds, IntentDoc v2
# measures 165,514,483 B at the pinned vocabulary, and the next power of two at or above twice that is
# 536,870,912. limits-freeze.json stays the frozen S0 record; every other limit is unchanged.
RELEASE = dict(RELEASE_S0, tracked_temporary_bytes=536_870_912)

# The trie charge the S1 converter adds to tracked bytes, which count model v2 does not model. Measured by the
# builder (brunel-s1-build.md, fix round 1: 79,396,816 B summed, 79,394,320 B by tracemalloc at the pinned
# vocabulary) and as the converter's tracked count minus count model v2's for IntentDoc v2 and G1 (79,419,072 and
# 79,407,744 B pinned; 60,280 and 48,952 B at the byte vocabulary). The tracked pairs keep a margin well above
# the spread between schemas.
TRIE_CHARGE = {"byte": 55_000, "pinned": 79_410_000}

# Release limits bind only schemas that contain an array; injected limits bind every schema (conductor ruling,
# brief-s1-fix-round1-rulings.md item 1).
PAIR_TARGET = {"R_MIN_ROOT": "minItems", "R_MIN_INNER": "minItems", "R_DEPTH": "recursion_depth",
               "R_NFA": "nfa_states", "R_DFA": "dfa_states", "R_TRANS": "token_transitions",
               "R_SECTION": "serialized_mask_bytes", "R_MEM": "tracked_temporary_bytes"}

# The array's location in a minItems diagnostic: an RFC 6901 JSON pointer into the schema (plan §4.1 l.424
# "names the array's JSON pointer"; pinned by the coverage auditor's ruling, S0a fold 7). A pointer is
# matched whole: it may not continue into a longer pointer, so the root pointer does not match the inner one.
ROOT_POINTERS = ("/properties/intents",)
INNER_POINTERS = ("/properties/intents/items/properties/conditions",)


def names_pointer(message: str, pointers) -> bool:
    return any(re.search(re.escape(p) + r"(?![A-Za-z0-9_~/])", message) for p in pointers)


def fresh(schema=INTENTDOC):
    return copy.deepcopy(schema)


# ------------------------------------------------------------------ IntentDoc v2 values

def atom(role="start", entity="beacon", state="lit", cmp="none", value="-", negated=False, watch="auto"):
    return {"role": role, "entity": entity, "state": state, "cmp": cmp, "value": value,
            "negated": negated, "watch": watch}


def intent(action="fire", subject="alpha wing", target="north gate", conditions=(), dep=False):
    return {"action": action, "subject": subject, "target": target, "conditions": list(conditions),
            "depends_on_previous": dep}


def doc(*intents):
    return {"intents": list(intents)}


def atoms(n, start=0):
    roles = ("start", "stop", "during", "start_any")
    cmps = (("none", "-"), ("below", "30"), ("at_least", "2"), ("reaches", "full"))
    out = []
    for i in range(start, start + n):
        c, v = cmps[i % len(cmps)]
        out.append(atom(role=roles[i % 3], entity=f"unit {i}", state="health" if c != "none" else "visible",
                        cmp=c, value=v, negated=bool(i % 2), watch=("auto", "reuse", "new")[i % 3]))
    return out


def intents(n, conditions_each=0):
    acts = ("fire", "move", "scan", "hold", "wait")
    return [intent(action=acts[i % 5], subject=f"squad {i}", target=f"point {i}",
                   conditions=atoms(conditions_each, i), dep=bool(i)) for i in range(n)]


# ------------------------------------------------------------------ schema nodes (§6 row 2)

ANNOTATIONS = frozenset({"title", "description", "default", "examples", "deprecated", "readOnly", "writeOnly",
                         "$comment"})
ROOT_ONLY = frozenset({"$schema", "$id"})
KIND_ALLOWED = {"enum": {"enum", "type"}, "boolean": {"type"}, "string": {"type"},
                "object": {"type", "properties", "required", "additionalProperties"},
                "array": {"type", "items", "minItems"}}
NODES = {  # location -> (path from the root, node kind)
    "root": ((), "object"),
    "intents": (("properties", "intents"), "array"),
    "intent": (("properties", "intents", "items"), "object"),
    "action": (("properties", "intents", "items", "properties", "action"), "enum"),
    "subject": (("properties", "intents", "items", "properties", "subject"), "string"),
    "target": (("properties", "intents", "items", "properties", "target"), "string"),
    "conditions": (("properties", "intents", "items", "properties", "conditions"), "array"),
    "atom": (("properties", "intents", "items", "properties", "conditions", "items"), "object"),
    "role": (("properties", "intents", "items", "properties", "conditions", "items", "properties", "role"), "enum"),
    "entity": (("properties", "intents", "items", "properties", "conditions", "items", "properties", "entity"), "string"),
    "state": (("properties", "intents", "items", "properties", "conditions", "items", "properties", "state"), "string"),
    "cmp": (("properties", "intents", "items", "properties", "conditions", "items", "properties", "cmp"), "enum"),
    "value": (("properties", "intents", "items", "properties", "conditions", "items", "properties", "value"), "string"),
    "negated": (("properties", "intents", "items", "properties", "conditions", "items", "properties", "negated"), "boolean"),
    "watch": (("properties", "intents", "items", "properties", "conditions", "items", "properties", "watch"), "enum"),
    "depends_on_previous": (("properties", "intents", "items", "properties", "depends_on_previous"), "boolean"),
}


def node(schema, location):
    current = schema
    for step in NODES[location][0]:
        current = current[step]
    return current


def allowed(location):
    kind = NODES[location][1]
    return KIND_ALLOWED[kind] | ANNOTATIONS | (ROOT_ONLY if location == "root" else set())


def with_min(root=1, inner=0):
    s = fresh()
    node(s, "intents")["minItems"] = root
    node(s, "conditions")["minItems"] = inner
    return s


def deepen_value(levels):
    """IntentDoc v2 with the atom's `value` leaf replaced by `levels` nested objects ending in a string.
    Depth (count model v2's definition: leaf 1, object or array 1 + child) is 6 + levels."""
    s = fresh()
    leaf = {"type": "string"}
    for i in range(levels):
        leaf = {"type": "object", "properties": {f"v{i}": leaf}, "required": [f"v{i}"],
                "additionalProperties": False}
    node(s, "atom")["properties"]["value"] = leaf
    return s


def deep_value(levels, text="30"):
    v = text
    for i in range(levels):
        v = {f"v{i}": v}
    return v


def obj(props):
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def triple_array(extra_object=False):
    """Three arrays nested in one another: depth 8, or 9 with one more object around the leaf."""
    leaf = {"type": "string", "enum": ["x"]}
    if extra_object:
        leaf = obj({"e": leaf})
    return obj({"a": {"type": "array", "minItems": 1, "items": obj({"b": {"type": "array", "minItems": 0,
                "items": obj({"c": {"type": "array", "minItems": 2, "items": obj({"d": leaf})}})}})}})


def in_required_order(schema):
    """resource-pairs-v2.json was written with sort_keys=True, which re-sorted every `properties` mapping
    while `required` kept the construction order. For R_SECTION (a pad key `f...` after `s00`..`s20`) the
    stored order differs from `required`, so the stored schema is not the schema that was counted, and the
    converter refuses it as reordered. The counted schema is recovered by ordering each object's properties
    by its `required` list; test_cv2_reference checks that this reproduces the frozen counts exactly."""
    if isinstance(schema, dict):
        out = {k: in_required_order(v) for k, v in schema.items()}
        if out.get("type") == "object" and isinstance(out.get("required"), list):
            out["properties"] = {k: out["properties"][k] for k in out["required"]}
        return out
    if isinstance(schema, list):
        return [in_required_order(v) for v in schema]
    return schema


def pair(name):
    for p in RESOURCE_PAIRS["pairs"]:
        if p["pair"] == name:
            p = copy.deepcopy(p)
            for side in ("under_schema", "over_schema"):
                if side in p:
                    p[side] = in_required_order(p[side])
            return p
    raise KeyError(name)


def stored_order_defects():
    """(pair, side) whose stored schema has an object with properties out of `required` order."""
    found = []

    def scan(n):
        if isinstance(n, dict):
            if n.get("type") == "object" and list(n.get("properties", {})) != n.get("required"):
                return True
            return any(scan(v) for v in n.values())
        return isinstance(n, list) and any(scan(v) for v in n)
    for p in RESOURCE_PAIRS["pairs"]:
        for side in ("under_schema", "over_schema"):
            if side in p and scan(p[side]):
                found.append((p["pair"], side))
    return found


def model_counts_to_contract(counts, vocab="byte"):
    """count_model_v2's keys to the contract's LIMIT_KEYS, with the trie charge added to tracked bytes for the
    vocabulary counted ("byte" or "pinned"; see TRIE_CHARGE)."""
    out = {key: counts[FREEZE_KEY[key]] for key in LIMIT_KEYS}
    out["tracked_temporary_bytes"] += TRIE_CHARGE[vocab]
    return out


def diagnostic_regex(key):
    return re.escape(DIAGNOSTICS[key])


def other_diagnostics(key):
    return [d for k, d in DIAGNOSTICS.items() if k != key]

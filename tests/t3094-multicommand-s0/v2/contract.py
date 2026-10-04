"""T-3103 S0 frozen IntentDoc v2 / G1 contract reference (plan rev 15 s1, s1.1, s0.1, s5.1 rule 1).

Written from the plan text; imports no converter, plugin or product code. It provides:
- strict parsing, which refuses duplicate keys, non-JSON constants and a non-object root;
- the shape check against the frozen schemas, covering key order, enums, types, minima and meaningful strings;
- the s1.1 semantic rules on exact bytes, before C or any normalization;
- the product predicate as the grader applies it (s5.1 rule 1): accepting walk at stop, completion stop,
  strict parse, shape, then semantic rules;
- the canonical serialization the schema DFA walks (compact separators, frozen key order).
"""
from __future__ import annotations

import json

ACTIONS = ("fire", "move", "scan", "hold", "wait")
ROLES = ("start", "start_any", "stop", "during")
CMPS = ("none", "below", "above", "at_most", "at_least", "reaches")
WATCHES = ("auto", "reuse", "new")
INTENT_KEYS = ("action", "subject", "target", "conditions", "depends_on_previous")
ATOM_KEYS = ("role", "entity", "state", "cmp", "value", "negated", "watch")
G1_KEYS = ("action", "subject", "target", "condition", "depends_on_previous")
G1G_KEYS = ("action", "subject", "target", "gated", "condition", "depends_on_previous")
SENTINEL = "-"
# The converter's insignificant bytes: a string value cannot close on these alone (plan s2).
INSIGNIFICANT = frozenset("{}[],: \t\r\n")
PLACEHOLDER_ATOM = {"role": "start", "entity": "no condition", "state": "no condition", "cmp": "none",
                    "value": "-", "negated": False, "watch": "auto"}


class Refusal(ValueError):
    def __init__(self, status: str, detail: str = ""):
        super().__init__(f"{status}: {detail}")
        self.status, self.detail = status, detail


class Pairs(list):
    """A parsed JSON object as its ordered (key, value) pairs; distinct from a JSON array."""


def _pairs(pairs):
    keys = [k for k, _ in pairs]
    if len(set(keys)) != len(keys):
        raise Refusal("InvalidDocument", "duplicate key")
    return Pairs(pairs)               # keep order: key order is part of the contract


def _const(name):
    raise Refusal("InvalidDocument", f"non-JSON constant {name}")


def parse(raw: bytes):
    try:
        text = raw.decode("utf-8")
        return json.loads(text, object_pairs_hook=_pairs, parse_constant=_const)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Refusal("InvalidDocument", str(error)) from None


def _obj(node, keys, where):
    if not isinstance(node, Pairs) or tuple(k for k, _ in node) != keys:
        raise Refusal("InvalidDocument", f"{where}: keys or order")
    return dict(node)


def _meaningful(value, where):
    if not isinstance(value, str) or not any(ch not in INSIGNIFICANT for ch in value):
        raise Refusal("InvalidDocument", f"{where}: empty or insignificant string")


def _atom(node, where):
    a = _obj(node, ATOM_KEYS, where)
    if a["role"] not in ROLES or a["cmp"] not in CMPS or a["watch"] not in WATCHES:
        raise Refusal("InvalidDocument", f"{where}: enum")
    for f in ("entity", "state", "value"):
        _meaningful(a[f], f"{where}.{f}")
    if not isinstance(a["negated"], bool):
        raise Refusal("InvalidDocument", f"{where}.negated")
    return a


def _intent(node, keys, where):
    it = _obj(node, keys, where)
    if it["action"] not in ACTIONS:
        raise Refusal("InvalidDocument", f"{where}.action")
    for f in ("subject", "target"):
        _meaningful(it[f], f"{where}.{f}")
    if not isinstance(it["depends_on_previous"], bool):
        raise Refusal("InvalidDocument", f"{where}.depends_on_previous")
    if "gated" in it and not isinstance(it["gated"], bool):
        raise Refusal("InvalidDocument", f"{where}.gated")
    if "conditions" in it:
        if not isinstance(it["conditions"], list) or isinstance(it["conditions"], Pairs):
            raise Refusal("InvalidDocument", f"{where}.conditions")
        it["conditions"] = [_atom(a, f"{where}.conditions[{i}]") for i, a in enumerate(it["conditions"])]
    if "condition" in it:
        it["condition"] = _atom(it["condition"], f"{where}.condition")
    return it


def shape(parsed, mode: str = "v2") -> dict:
    """Shape check against the frozen schema. mode: 'v2', 'g1' or 'g1g'. Returns plain dicts."""
    if mode == "v2":
        root = _obj(parsed, ("intents",), "$")
        items = root["intents"]
        if not isinstance(items, list) or len(items) < 1:
            raise Refusal("InvalidDocument", "$.intents minimum")
        return {"intents": [_intent(x, INTENT_KEYS, f"$.intents[{i}]") for i, x in enumerate(items)]}
    keys = G1_KEYS if mode == "g1" else G1G_KEYS
    root = _obj(parsed, ("intent",), "$")
    return {"intent": _intent(root["intent"], keys, "$.intent")}


def semantic_rules(doc: dict) -> None:
    """Plan s1.1 rules (a)-(c) on exact bytes. A violation is InvalidConditionAtom."""
    intents = doc["intents"] if "intents" in doc else [doc["intent"]]
    for i, it in enumerate(intents):
        atoms = it.get("conditions", [it["condition"]] if "condition" in it else [])
        for j, a in enumerate(atoms):
            if (a["cmp"] == "none") != (a["value"] == SENTINEL):
                raise Refusal("InvalidConditionAtom", f"intent {i} atom {j}: rule (a) cmp/value")
            if a["entity"] == SENTINEL or a["state"] == SENTINEL:
                raise Refusal("InvalidConditionAtom", f"intent {i} atom {j}: rule (c) sentinel entity/state")
        n_any = sum(a["role"] == "start_any" for a in atoms)
        if n_any == 1:
            raise Refusal("InvalidConditionAtom", f"intent {i}: rule (b) lone start_any")


def product_predicate(raw: bytes, *, accepting: bool, stop: str, mode: str = "v2") -> dict:
    """Grader's implementation of the product acceptance predicate (s5.1 rule 1).

    stop: 'Complete', 'Budget', 'DeadEnd' or 'StopToken'. Returns the shaped document or raises Refusal."""
    if stop == "Budget":
        raise Refusal("BudgetStop")
    if stop == "DeadEnd":
        raise Refusal("SchemaDeadEnd")
    if stop != "Complete" or not accepting:
        raise Refusal("NonacceptingStop")
    doc = shape(parse(raw), mode)
    semantic_rules(doc)
    return doc


def canonical_bytes(doc: dict) -> bytes:
    """The canonical serialization the schema DFA walks: compact, frozen key order, UTF-8 unescaped."""
    return json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def g1_to_v2(doc: dict) -> dict:
    """The explicit, code-reviewed G1 (or G1g) -> one-intent IntentDoc v2 mapping for the gate grader."""
    it = doc["intent"]
    conditions = [] if it.get("gated") is False else [it["condition"]]
    return {"intents": [{"action": it["action"], "subject": it["subject"], "target": it["target"],
                         "conditions": conditions, "depends_on_previous": it["depends_on_previous"]}]}

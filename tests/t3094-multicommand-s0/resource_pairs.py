"""Independent S0 guard inputs and a finite-language reference automaton.

The reference counts its own epsilon NFA, subset DFA, byte-token edges, SCM1
bytes and explicit tracked-allocation model. It never reads converter counters.
Its chosen numeric boundaries are provisional: S1 reconciles the production
representation and adjusts the injected target limit before claiming green.
"""
from __future__ import annotations

import copy
import json
from collections import deque
from pathlib import Path

BASE = json.loads((Path(__file__).parent / "command.schema.json").read_text(encoding="utf-8"))
DIAGNOSTICS = {
    "minItems": "minItems limit",
    "recursion_depth": "recursion depth limit",
    "nfa_states": "NFA state limit",
    "dfa_states": "DFA state limit",
    "token_transitions": "token transition limit",
    "serialized_mask_bytes": "serialized mask byte limit",
    "tracked_temporary_bytes": "tracked temporary memory limit",
}


def _finite(node):
    """Resource fixtures avoid unbounded content cycles and use finite enums."""
    if node.get("type") == "string" and "enum" not in node:
        node["enum"] = ["sample"]
    for child in node.get("properties", {}).values():
        _finite(child)
    if isinstance(node.get("items"), dict):
        _finite(node["items"])


def _ast(node):
    if "enum" in node:
        return ("alt", [json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                        for value in node["enum"]])
    kind = node["type"]
    if kind == "boolean":
        return ("alt", ["true", "false"])
    if kind == "object":
        children = ["{"]
        for index, (name, value) in enumerate(node["properties"].items()):
            if index:
                children.append(",")
            children.extend((json.dumps(name, ensure_ascii=False) + ":", _ast(value)))
        children.append("}")
        return ("seq", children)
    if kind == "array":
        item = _ast(node["items"])
        minimum = node.get("minItems", 0)
        repeat = ("star", ("seq", [",", item]))
        if minimum:
            body = ("seq", [item] + [("seq", [",", item]) for _ in range(minimum - 1)] + [repeat])
        else:
            body = ("alt", ["", ("seq", [item, repeat])])
        return ("seq", ["[", body, "]"])
    raise ValueError(f"nonfinite fixture node: {node}")


def model_counts(schema, vocab_size=256):
    """Thompson epsilon-NFA followed by subset construction, independent of S1."""
    epsilon = []
    byte_edges = []

    def new():
        epsilon.append(set())
        byte_edges.append({})
        return len(epsilon) - 1

    def make(expr):
        start, end = new(), new()
        if isinstance(expr, str):
            position = start
            for byte in expr.encode("utf-8"):
                following = new()
                byte_edges[position].setdefault(byte, set()).add(following)
                position = following
            epsilon[position].add(end)
        elif expr[0] == "seq":
            position = start
            for child in expr[1]:
                a, b = make(child)
                epsilon[position].add(a)
                position = b
            epsilon[position].add(end)
        elif expr[0] == "alt":
            for child in expr[1]:
                a, b = make(child)
                epsilon[start].add(a)
                epsilon[b].add(end)
        elif expr[0] == "star":
            a, b = make(expr[1])
            epsilon[start].update((a, end))
            epsilon[b].update((a, end))
        return start, end

    start, end = make(_ast(schema))

    def closure(nodes):
        found = set(nodes)
        stack = list(nodes)
        while stack:
            for target in epsilon[stack.pop()]:
                if target not in found:
                    found.add(target)
                    stack.append(target)
        return frozenset(found)

    first = closure({start})
    seen = {first: 0}
    queue = deque([first])
    transitions = 0
    accepting = 0
    while queue:
        current = queue.popleft()
        if end in current:
            accepting += 1
        moves = {}
        for state in current:
            for byte, targets in byte_edges[state].items():
                moves.setdefault(byte, set()).update(targets)
        transitions += len(moves)
        for targets in moves.values():
            next_set = closure(targets)
            if next_set not in seen:
                seen[next_set] = len(seen)
                queue.append(next_set)
    nodes = len(epsilon)
    epsilon_count = sum(map(len, epsilon))
    byte_count = sum(len(targets) for row in byte_edges for targets in row.values())
    states = len(seen)
    page_width = (vocab_size + 7) // 8
    mask_bytes = 24 + 56 + len("commands") + states * page_width + (states + 1) * 4 + accepting * 4 + transitions * 8
    # Explicit reservation model for nodes, graph edges, DFA frontier, byte trie,
    # CSR transition staging and serialized masks. S1 must name its own tracked
    # allocations and reconcile; process RSS is a separate observation.
    memory = nodes * 64 + (epsilon_count + byte_count) * 16 + states * 48 + 257 * 32 + transitions * 16 + mask_bytes
    return {"minItems": schema["properties"]["commands"].get("minItems", 0),
            "recursion_depth": _depth(schema), "nfa_states": nodes, "dfa_states": states,
            "token_transitions": transitions, "serialized_mask_bytes": mask_bytes,
            "tracked_temporary_bytes": memory}


def _depth(node):
    children = list(node.get("properties", {}).values())
    if isinstance(node.get("items"), dict):
        children.append(node["items"])
    return 1 + max((_depth(child) for child in children), default=0)


def _finish(under, over, guard):
    _finite(under)
    _finite(over)
    below, above = model_counts(under), model_counts(over)
    assert below[guard] < above[guard], (guard, below, above)
    return under, over, below, above


def _simple_enum(schema, values):
    item = schema["properties"]["commands"]["items"]
    item["properties"]["subject"] = {"type": "string", "enum": values}
    item["properties"]["target"] = {"type": "string", "enum": ["ridge"]}
    item["properties"]["trigger"] = {"type": "string", "enum": ["signal"]}


def pair(name):
    under, over = copy.deepcopy(BASE), copy.deepcopy(BASE)
    if name == "minItems":
        under["properties"]["commands"]["minItems"] = 1
        over["properties"]["commands"]["minItems"] = 2
        return _finish(under, over, name)
    if name == "recursion_depth":
        under["properties"]["commands"]["items"]["properties"]["subject"] = {
            "type": "object", "properties": {"name": {"type": "string"}},
            "required": ["name"], "additionalProperties": False}
        over = copy.deepcopy(under)
        over["properties"]["commands"]["items"]["properties"]["subject"]["properties"]["name"] = {
            "type": "object", "properties": {"value": {"type": "string"}},
            "required": ["value"], "additionalProperties": False}
        return _finish(under, over, name)
    if name == "nfa_states":
        a = ["alpha"]
        b = ["alpha", "bravo-long-prefix"]
        _simple_enum(under, a)
        _simple_enum(over, b)
        return _finish(under, over, name)
    if name == "dfa_states":
        a = ["a0", "a1"]
        b = ["a0", "a1", "a234567"]
        _simple_enum(under, a)
        _simple_enum(over, b)
        return _finish(under, over, name)
    if name == "token_transitions":
        a = ["alpha"]
        b = ["alpha", "bravo", "charlie", "delta"]
        _simple_enum(under, a)
        _simple_enum(over, b)
        return _finish(under, over, name)
    if name == "serialized_mask_bytes":
        a = ["alpha"]
        b = ["alpha", "bravo", "charlie"]
        _simple_enum(under, a)
        _simple_enum(over, b)
        return _finish(under, over, name)
    if name == "tracked_temporary_bytes":
        a = ["alpha"]
        b = ["alpha", "bravo", "charlie", "delta", "echo"]
        _simple_enum(under, a)
        _simple_enum(over, b)
        return _finish(under, over, name)
    raise KeyError(name)

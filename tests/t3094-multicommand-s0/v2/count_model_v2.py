"""T-3103 S0 count model v2 (plan rev 15 s4.1; replaces T-3094-RESOURCE). Counts a schema's resources
at the pinned real vocabulary, independently of the converter and of the spike probe: it imports
neither. It is written from the published constructions:
- Literal groups and enums/booleans are epsilon-free byte chains. Alternatives share their start node
  and a fresh `following` node.
- A string leaf is the two-region U/M automaton of the converter contract (UTF-8 validity with
  overlong, surrogate and range exclusions, short and \\uXXXX escapes, value-level closure on
  insignificant bytes).
- A fixed object is '{', keys, ',' and '}'.
- An array with minItems >= 1 has its required copies, with ',' looping back into the last copy. An
  array with minItems 0 has a direct '[]' path, a first-item copy and a looping repeat copy (the
  witness's construction, plan s3 reading 9).
- Subset construction gives the character DFA.
- The token DFA comes from an own trie of the vocabulary's raw bytes (first id per spelling, special
  ids excluded) under the asymmetric boundary rule: opening a string value only on a token's first
  byte; closing unrestricted.

Resources per s4.1: minItems per array, recursion depth, NFA nodes, DFA states, token transitions,
SCM1 section bytes (frozen formula) and tracked temporary bytes (frozen allocation model, T-3094
resource_pairs.py:137-139).

Usage:
  py -3 count_model_v2.py count --schema <json> --name <section name> [--tokenizer <tokenizer.json> --vocab-size N]
  py -3 count_model_v2.py reconcile --tokenizer <tokenizer.json> --vocab-size 151936 --witness <witness_real.json>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
INSIGNIFICANT = frozenset(b"{}[],: \t\r\n")
SHORT_ESCAPES = {'"': 0x22, "\\": 0x5C, "/": 0x2F, "b": 0x08, "f": 0x0C, "n": 0x0A, "r": 0x0D, "t": 0x09}
HEX = "0123456789abcdefABCDEF"
RELEASE_LIMITS = {"minItems": 8, "recursion_depth": 8, "nfa_states": 4096, "dfa_states": 4096,
                  "token_transitions": 8388608, "section_bytes": 134217728, "tracked_bytes": 268435456}


# ------------------------------------------------------------------ byte NFA

class NFA:
    def __init__(self):
        self.edges = [{}]
        self.content = set()

    def node(self, content=False):
        self.edges.append({})
        n = len(self.edges) - 1
        if content:
            self.content.add(n)
        return n

    def add(self, a, byte, b):
        self.edges[a].setdefault(byte, set()).add(b)

    def literal(self, start, text, end=None):
        raw = text.encode("utf-8")
        end = self.node() if end is None else end
        cur = start
        for i, b in enumerate(raw):
            nxt = end if i == len(raw) - 1 else self.node()
            self.add(cur, b, nxt)
            cur = nxt
        return end

    def alternatives(self, start, alts):
        end = self.node()
        for a in alts:
            self.literal(start, a, end)
        return end

    def string_leaf(self, start):
        c = lambda: self.node(True)  # noqa: E731
        rng = lambda a, lo, hi, b: [self.add(a, x, b) for x in range(lo, hi + 1)]  # noqa: E731
        m = c()
        l2 = c()
        rng(m, 0xC2, 0xDF, l2)
        rng(l2, 0x80, 0xBF, m)
        l3a, l3b, l3c, mid3 = c(), c(), c(), c()
        rng(m, 0xE0, 0xE0, l3a)
        rng(m, 0xE1, 0xEC, l3b)
        rng(m, 0xED, 0xED, l3c)
        rng(m, 0xEE, 0xEF, l3b)
        rng(l3a, 0xA0, 0xBF, mid3)
        rng(l3b, 0x80, 0xBF, mid3)
        rng(l3c, 0x80, 0x9F, mid3)
        rng(mid3, 0x80, 0xBF, m)
        l4a, l4b, l4c, m4a, m4b = c(), c(), c(), c(), c()
        rng(m, 0xF0, 0xF0, l4a)
        rng(m, 0xF1, 0xF3, l4b)
        rng(m, 0xF4, 0xF4, l4c)
        rng(l4a, 0x90, 0xBF, m4a)
        rng(l4b, 0x80, 0xBF, m4a)
        rng(l4c, 0x80, 0x8F, m4a)
        rng(m4a, 0x80, 0xBF, m4b)
        rng(m4b, 0x80, 0xBF, m)
        for x in range(0x20, 0x80):
            if x not in (0x22, 0x5C):
                self.add(m, x, m)
        esc = c()
        self.add(m, 0x5C, esc)
        for ch in SHORT_ESCAPES:
            self.add(esc, ord(ch), m)
        hexes = [c() for _ in range(4)]
        self.add(esc, ord("u"), hexes[0])
        for i, h in enumerate(hexes):
            for d in HEX:
                self.add(h, ord(d), hexes[i + 1] if i + 1 < 4 else m)
        u = c()
        self.add(start, 0x22, u)
        for x in range(0x20, 0x80):
            if x not in (0x22, 0x5C):
                self.add(u, x, u if x in INSIGNIFICANT else m)
        for lo, hi, tgt in ((0xC2, 0xDF, l2), (0xE0, 0xE0, l3a), (0xE1, 0xEC, l3b), (0xED, 0xED, l3c),
                            (0xEE, 0xEF, l3b), (0xF0, 0xF0, l4a), (0xF1, 0xF3, l4b), (0xF4, 0xF4, l4c)):
            rng(u, lo, hi, tgt)
        uesc = c()
        self.add(u, 0x5C, uesc)
        for ch, dec in SHORT_ESCAPES.items():
            self.add(uesc, ord(ch), u if dec in INSIGNIFICANT else m)
        insig_hex = tuple(f"{cp:04x}" for cp in sorted(INSIGNIFICANT))
        generic = {}

        def remainder(k):
            if k == 0:
                return m
            if k not in generic:
                s = c()
                generic[k] = s
                t = remainder(k - 1)
                for d in HEX:
                    self.add(s, ord(d), t)
            return generic[k]
        prefixes = {}

        def prefix(p):
            if p in prefixes:
                return prefixes[p]
            s = c()
            prefixes[p] = s
            for d in HEX:
                cand = p + d.lower()
                if len(cand) == 4:
                    t = u if int(cand, 16) in INSIGNIFICANT else m
                elif any(v.startswith(cand) for v in insig_hex):
                    t = prefix(cand)
                else:
                    t = remainder(4 - len(cand))
                self.add(s, ord(d), t)
            return s
        self.add(uesc, ord("u"), prefix(""))
        following = self.node()
        self.add(m, 0x22, following)
        return following

    def schema(self, s, start, path="$", arrays=None):
        t = s.get("type")
        if "enum" in s:
            return self.alternatives(start, [json.dumps(v, ensure_ascii=False) for v in s["enum"]])
        if t == "boolean":
            return self.alternatives(start, ["true", "false"])
        if t == "string":
            return self.string_leaf(start)
        if t == "object":
            cur = self.literal(start, "{")
            for i, (k, child) in enumerate(s["properties"].items()):
                if i:
                    cur = self.literal(cur, ",")
                cur = self.literal(cur, json.dumps(k, ensure_ascii=False) + ":")
                cur = self.schema(child, cur, f"{path}.{k}", arrays)
            return self.literal(cur, "}")
        if t == "array":
            minimum = s.get("minItems", 0)
            if arrays is not None:
                arrays[path] = minimum
            following = self.node()
            cur = self.literal(start, "[")
            item = s["items"]
            if minimum == 0:
                self.literal(cur, "]", following)
                first_end = self.schema(item, cur, path + "[]", arrays)
                self.literal(first_end, "]", following)
                rep = self.node()
                self.literal(first_end, ",", rep)
                rep_end = self.schema(item, rep, path + "[]", arrays)
                self.literal(rep_end, "]", following)
                self.literal(rep_end, ",", rep)
                return following
            rep = cur
            for i in range(minimum):
                if i:
                    cur = self.literal(cur, ",")
                rep = cur
                cur = self.schema(item, cur, path + "[]", arrays)
            self.literal(cur, "]", following)
            self.literal(cur, ",", rep)
            return following
        raise ValueError(f"unsupported construct at {path}")


def depth(s):
    t = s.get("type")
    if t == "object":
        return 1 + max((depth(c) for c in s["properties"].values()), default=0)
    if t == "array":
        return 1 + depth(s["items"])
    return 1


def char_dfa(nfa, accept):
    start = frozenset({0})
    ids, dfa, acc, content = {start: 0}, [{}], set(), set()
    q = deque([start])
    while q:
        cur = q.popleft()
        sid = ids[cur]
        if accept in cur:
            acc.add(sid)
        if cur & nfa.content:
            content.add(sid)
        moves = {}
        for n in cur:
            for b, ts in nfa.edges[n].items():
                moves.setdefault(b, set()).update(ts)
        row = {}
        for b in sorted(moves):
            ts = frozenset(moves[b])
            if ts not in ids:
                ids[ts] = len(dfa)
                dfa.append({})
                q.append(ts)
            row[b] = ids[ts]
        dfa[sid] = row
    return dfa, acc, content


# ------------------------------------------------------------------ vocabulary and token walk

def _byte_decoder():
    """GPT-2 byte-level BPE: the printable stand-in character for each byte, inverted."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def load_vocab(tokenizer_json, vocab_size):
    tj = json.loads(Path(tokenizer_json).read_text(encoding="utf-8"))
    dec = _byte_decoder()
    pieces = [b""] * vocab_size
    for tok, idx in tj["model"]["vocab"].items():
        if idx < vocab_size:
            pieces[idx] = bytes(dec[ch] for ch in tok)
    special = [a["id"] for a in tj.get("added_tokens", [])]
    for i in special:
        if i < vocab_size:
            pieces[i] = b""
    return pieces, len(special)


def build_trie(pieces):
    root = {}
    for tid, p in enumerate(pieces):
        if not p:
            continue
        node = root
        for b in p:
            node = node.setdefault(b, {})
        node.setdefault(-1, tid)          # first (lowest) id spelling this piece
    return root


def token_targets(dfa, state, trie, content):
    targets = {}
    stack = [(trie, state, 0, False)]
    while stack:
        node, cur, d, stop = stack.pop()
        tid = node.get(-1)
        if tid is not None and d > 0:
            targets[tid] = cur
        if stop:
            continue
        row = dfa[cur]
        cur_content = cur in content
        for b, child in node.items():
            if b == -1:
                continue
            nxt = row.get(b)
            if nxt is None:
                continue
            opening = (not cur_content) and nxt in content
            if opening and d != 0:
                continue
            stack.append((child, nxt, d + 1, opening))
    return targets


def count(schema, name, pieces=None, vocab_size=256):
    nfa = NFA()
    arrays = {}
    accept = nfa.schema(schema, 0, "$", arrays)
    dfa, acc, content = char_dfa(nfa, accept)
    if pieces is None:
        pieces = [bytes([i]) for i in range(256)]
        vocab_size = 256
    trie = build_trie(pieces)
    ids, q, trans, accepting = {0: 0}, deque([0]), 0, 0
    while q:
        cur = q.popleft()
        if cur in acc:
            accepting += 1
        tg = token_targets(dfa, cur, trie, content)
        trans += len(tg)
        for t in tg.values():
            if t not in ids:
                ids[t] = len(ids)
                q.append(t)
    states = len(ids)
    page = (vocab_size + 7) // 8
    section = 24 + 56 + len(name.encode()) + states * page + (states + 1) * 4 + accepting * 4 + trans * 8
    nodes = len(nfa.edges)
    byte_edges = sum(len(ts) for row in nfa.edges for ts in row.values())
    tracked = nodes * 64 + byte_edges * 16 + states * 48 + 257 * 32 + trans * 16 + section
    return {"minItems": arrays, "recursion_depth": depth(schema), "nfa_states": nodes, "nfa_byte_edges": byte_edges,
            "char_dfa_states": len(dfa), "dfa_states": states, "accepting": accepting, "token_transitions": trans,
            "section_bytes": section, "tracked_bytes": tracked}


def crossings(counts, limits=RELEASE_LIMITS):
    out = []
    if any(v > limits["minItems"] for v in counts["minItems"].values()):
        out.append("minItems")
    for k in ("recursion_depth", "nfa_states", "dfa_states", "token_transitions", "section_bytes", "tracked_bytes"):
        if counts[k] > limits[k]:
            out.append(k)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("count", "reconcile"))
    ap.add_argument("--schema")
    ap.add_argument("--name", default="schema")
    ap.add_argument("--tokenizer")
    ap.add_argument("--vocab-size", type=int, default=151936)
    ap.add_argument("--witness")
    args = ap.parse_args(argv)
    pieces = None
    if args.tokenizer:
        pieces, n_special = load_vocab(args.tokenizer, args.vocab_size)
    if args.cmd == "count":
        t0 = time.time()
        out = count(json.loads(Path(args.schema).read_text(encoding="utf-8")), args.name, pieces, args.vocab_size)
        out["seconds"] = round(time.time() - t0, 1)
        print(json.dumps(out, sort_keys=True))
        return 0
    witness = {r["schema"]: r for r in json.loads(Path(args.witness).read_text(encoding="utf-8"))["results"]}
    rows, ok = [], True
    for label, path, name in (("G1", "g1.schema.json", "intent_gate"), ("IntentDoc v2", "intentdoc-v2.schema.json", "intents")):
        mine = count(json.loads((HERE / path).read_text(encoding="utf-8")), name, pieces, args.vocab_size)
        w = witness[label]
        pairs = {"nfa_states": "nfa_nodes", "nfa_byte_edges": "nfa_byte_edges", "dfa_states": "token_dfa_states",
                 "accepting": "accepting_states", "token_transitions": "token_transitions",
                 "section_bytes": "scm1_section_bytes_frozen_formula", "tracked_bytes": "tracked_bytes_frozen_model",
                 "recursion_depth": "recursion_depth"}
        diff = {k: (mine[k], w[v]) for k, v in pairs.items() if mine[k] != w[v]}
        ok &= not diff
        rows.append({"schema": label, "count_model_v2": mine, "differs_from_witness": diff})
    print(json.dumps({"special_ids_excluded": n_special, "reconciliation": rows, "all_equal": ok}, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

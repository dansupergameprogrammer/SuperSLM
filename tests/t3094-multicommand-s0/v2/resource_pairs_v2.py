"""T-3103 S0 resource-limit pairs (plan rev 15 s4.1, D-SLM8087/8095/8104/8120), counted by count model v2
at the pinned real vocabulary against the provisional release limits.

For each guard there is an independently constructed under/over schema pair. The under schema sits at
the limit, admitted on every guard. The over schema crosses only its target guard, by one unit where the
construction allows it. A pair whose target cannot be crossed alone is declared injection-only with its
reason (R_MEM by D-SLM8120; R_DFA by finding MX-8 if the search confirms it).

Usage: py -3 resource_pairs_v2.py --tokenizer <tokenizer.json> [--vocab-size 151936] --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import count_model_v2 as cm  # noqa: E402

L = cm.RELEASE_LIMITS
LEAF = {"type": "string"}


def obj(props):
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def enum(values):
    return {"type": "string", "enum": values}


def min_root(n):
    return obj({"a": {"type": "array", "minItems": n, "items": enum(["x"])}})


def min_inner(n):
    return obj({"a": {"type": "array", "minItems": 1,
                      "items": obj({"b": {"type": "array", "minItems": n, "items": enum(["x"])}})}})


def nest(d):
    s = enum(["x"])
    for _ in range(d - 1):
        s = obj({"k": s})
    return s


def nfa_family(alternatives, prefix_len, pad):
    """Many alternatives sharing one long prefix: NFA nodes grow by prefix_len per alternative, while DFA
    states grow by one (the subset construction merges the shared prefix)."""
    p = "q" * prefix_len
    return obj({"k" + "p" * pad: enum([p + format(i, "03d") for i in range(alternatives)])})


def leaves(n, pad=0):
    props = {f"s{i:02d}": LEAF for i in range(n)}
    if pad:
        props["f" + "".join(chr(0x61 + (i % 26)) for i in range(pad))] = enum(["x"])
    return obj(props)


def run_pair(name, under, over, target, pieces, vocab_size, note=""):
    cu = cm.count(under, "pair", pieces, vocab_size)
    co = cm.count(over, "pair", pieces, vocab_size)
    xu, xo = cm.crossings(cu), cm.crossings(co)
    return {"pair": name, "target": target, "under": cu, "over": co, "under_crossings": xu, "over_crossings": xo,
            "isolated": xu == [] and xo == [target], "note": note, "under_schema": under, "over_schema": over}


def search_nfa(pieces, vocab_size):
    """Find a pad so the NFA family sits at exactly 4,096 nodes; the over schema adds one pad byte."""
    base = cm.count(nfa_family(100, 30, 0), "pair", pieces, vocab_size)["nfa_states"]
    pad = L["nfa_states"] - base
    assert pad >= 0, base
    return nfa_family(100, 30, pad), nfa_family(100, 30, pad + 1)


def search_leaves(target_key, pieces, vocab_size, pad=0):
    lo = 1
    while cm.count(leaves(lo + 1, pad), "pair", pieces, vocab_size)[target_key] <= L[target_key]:
        lo += 1
    return lo


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--vocab-size", type=int, default=151936)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    t0 = time.time()
    pieces, _ = cm.load_vocab(args.tokenizer, args.vocab_size)
    vs = args.vocab_size
    pairs = [run_pair("R_MIN_ROOT", min_root(8), min_root(9), "minItems", pieces, vs),
             run_pair("R_MIN_INNER", min_inner(8), min_inner(9), "minItems", pieces, vs),
             run_pair("R_DEPTH", nest(8), nest(9), "recursion_depth", pieces, vs)]
    u, o = search_nfa(pieces, vs)
    pairs.append(run_pair("R_NFA", u, o, "nfa_states", pieces, vs))
    # R_DFA: every construction in this subset determinizes to at most as many DFA states as NFA nodes,
    # so a 4,097-state DFA has at least 4,097 NFA nodes. Executed on the largest pure-literal family.
    k = 4096 - cm.count(obj({"k": enum(["x"])}), "pair", pieces, vs)["nfa_states"]
    dfa_under = obj({"k" + "z" * k: enum(["x"])})
    dfa_over = obj({"k" + "z" * (k + 1): enum(["x"])})
    pairs.append(run_pair("R_DFA", dfa_under, dfa_over, "dfa_states", pieces, vs,
                          note="pure literal family: DFA states equal NFA nodes here, so the over schema crosses "
                               "both state guards (finding MX-8: injection-only at release limits)"))
    n_trans = search_leaves("token_transitions", pieces, vs)
    pairs.append(run_pair("R_TRANS", leaves(n_trans), leaves(n_trans + 1), "token_transitions", pieces, vs))
    # R_SECTION: enough leaves to approach the transition budget, then literal padding to raise DFA states
    # (each state costs one 18,992-byte mask page) until the section crosses while transitions, states and
    # tracked memory stay admitted.
    base_leaves = n_trans
    lo, hi = 0, 3600
    while lo < hi:
        mid = (lo + hi + 1) // 2
        c = cm.count(leaves(base_leaves, mid), "pair", pieces, vs)
        if c["section_bytes"] <= L["section_bytes"] and not cm.crossings(c):
            lo = mid
        else:
            hi = mid - 1
    pairs.append(run_pair("R_SECTION", leaves(base_leaves, lo), leaves(base_leaves, lo + 1), "section_bytes",
                          pieces, vs))
    pairs.append({"pair": "R_MEM", "target": "tracked_bytes", "isolated": False, "injection_only": True,
                  "note": "D-SLM8120: tracked = 64N + 16E + 48S + 8,224 + 16T + section, and the memory limit "
                          "equals 16 x the transition limit + the section limit; with every other guard admitted, "
                          "tracked exceeds its limit only within about 0.006% of the transition limit at full "
                          "section bytes. Isolated by injected limits through the production guard."})
    for p in pairs:
        if p["pair"] == "R_DFA" and not p["isolated"]:
            p["injection_only"] = True
    out = {"limits": L, "pairs": pairs, "seconds": round(time.time() - t0, 1)}
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for p in pairs:
        print(json.dumps({k: p[k] for k in ("pair", "target", "isolated") if k in p} |
                         {"under_crossings": p.get("under_crossings"), "over_crossings": p.get("over_crossings")}))
    print(f"seconds {out['seconds']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

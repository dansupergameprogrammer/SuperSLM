# T-1421 independent defect population generator.
#
# Route independence is the whole point of this file. It enumerates mutation
# SITES mechanically, by syntactic operator class, over every translation unit
# on the decode arithmetic path -- with no knowledge of, and no reference to,
# what any oracle is expected to do about them. Selection is a seeded uniform
# shuffle over the full enumerated population, walked in order.
#
# Operator set (Offutt's sufficient set, plus statement deletion):
#   ROR  relational operator replacement
#   AOR  arithmetic operator replacement
#   SOR  shift operator replacement
#   SAR  shift amount +/- 1
#   CRP  integer constant replacement (c+1, c-1)
#   LCR  logical connector replacement
#   SDL  statement deletion (simple assignment statements)
#
# Scope: the five TUs that carry decode arithmetic. Declared before any site
# was enumerated and not revised afterwards.

import json
import random
import re
import sys

FILES = [
    "src/forward/forward_sites.cpp",
    "src/forward/checked_chain_funnel.cpp",
    "src/matmul.cpp",
    "src/intmath.cpp",
    "src/silu_lut.cpp",
]

SEED = 20260731


def code_mask(text):
    """True where the character is code (not comment, string or char literal)."""
    mask = [True] * len(text)
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                mask[k] = False
            i = j
        elif c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, j):
                mask[k] = False
            i = j
        elif c in "\"'":
            q = c
            j = i + 1
            while j < n:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == q:
                    j += 1
                    break
                j += 1
            for k in range(i, min(j, n)):
                mask[k] = False
            i = j
        else:
            i += 1
    return mask


def add(sites, kind, pos, old, new, text):
    line = text.count("\n", 0, pos) + 1
    sites.append({"kind": kind, "pos": pos, "old": old, "new": new, "line": line})


def enumerate_sites(path, text):
    mask = code_mask(text)
    sites = []

    def is_code(a, b):
        return all(mask[a:b])

    # ROR -- relational / equality operators.
    for m in re.finditer(r"(<=>)|(<<=|>>=)|(<<|>>)|(<=|>=|==|!=)|(<|>)", text):
        if not is_code(m.start(), m.end()):
            continue
        op = m.group(0)
        if op in ("<=>", "<<=", ">>=", "<<", ">>"):
            continue
        repl = {
            "<": ["<=", ">"],
            ">": [">=", "<"],
            "<=": ["<"],
            ">=": [">"],
            "==": ["!="],
            "!=": ["=="],
        }[op]
        # A bare < or > is very likely a template bracket; require whitespace or
        # an identifier/number on both sides in a comparison-looking context.
        if op in ("<", ">"):
            before = text[max(0, m.start() - 1) : m.start()]
            after = text[m.end() : m.end() + 1]
            if before in ("", "-", "<", ">") or after in ("", "<", ">"):
                continue
            # skip include-style and template-ish neighbours
            ctx = text[max(0, m.start() - 24) : m.start()]
            if re.search(r"(template|vector|std::|_t|int8_t|int32_t|int64_t|uint\w*)\s*$", ctx):
                continue
        for r in repl:
            add(sites, "ROR", m.start(), op, r, text)

    # SOR -- shift operator replacement.
    for m in re.finditer(r"(?<![<>])(<<|>>)(?![<>=])", text):
        if not is_code(m.start(), m.end()):
            continue
        op = m.group(0)
        add(sites, "SOR", m.start(), op, ">>" if op == "<<" else "<<", text)

    # SAR -- shift amount off by one.
    for m in re.finditer(r"(<<|>>)\s*(\d+)", text):
        if not is_code(m.start(), m.end()):
            continue
        amt = int(m.group(2))
        s = m.start(2)
        add(sites, "SAR", s, m.group(2), str(amt + 1), text)
        if amt > 0:
            add(sites, "SAR", s, m.group(2), str(amt - 1), text)

    # AOR -- binary arithmetic operator replacement.
    for m in re.finditer(r"(?<![+\-*/%<>=!&|^])([+\-*/%])(?![+\-*/%=>])", text):
        if not is_code(m.start(), m.end()):
            continue
        op = m.group(1)
        before = text[max(0, m.start() - 1) : m.start()]
        after = text[m.end() : m.end() + 1]
        if before.strip() == "" and after.strip() == "":
            pass
        # unary minus / pointer-star heuristics: require an identifier, digit or
        # closing bracket immediately before (ignoring one space).
        prev = text[: m.start()].rstrip()
        if not prev or prev[-1] not in ")]}_0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
            continue
        repl = {"+": "-", "-": "+", "*": "+", "/": "*", "%": "/"}[op]
        add(sites, "AOR", m.start(), op, repl, text)

    # LCR -- logical connectors.
    for m in re.finditer(r"(&&|\|\|)", text):
        if not is_code(m.start(), m.end()):
            continue
        op = m.group(0)
        add(sites, "LCR", m.start(), op, "||" if op == "&&" else "&&", text)

    # CRP -- integer literal replacement.
    for m in re.finditer(r"(?<![\w.])(\d+)(?![\w.])", text):
        if not is_code(m.start(), m.end()):
            continue
        lit = m.group(1)
        if len(lit) > 12:
            continue
        v = int(lit)
        add(sites, "CRP", m.start(), lit, str(v + 1), text)
        add(sites, "CRP", m.start(), lit, str(v - 1), text)

    # SDL -- delete a simple assignment or call statement.
    pat = (r"(?m)^[ \t]*(\*?[A-Za-z_][\w\.\[\]]*(?:->[\w\.\[\]]+)*)\s*"
           r"(=|\+=|-=|\*=|\|=|&=|<<=|>>=)\s*[^;{}]*;[ \t]*\r?$")
    for m in re.finditer(pat, text):
        if not is_code(m.start(), m.end()):
            continue
        head = m.group(1)
        if head in ("return", "if", "for", "while", "else", "case", "const", "static"):
            continue
        stmt = text[m.start() : m.end()].rstrip("\r")
        add(sites, "SDL", m.start(), stmt, " " * len(stmt), text)

    # SDL2 -- delete a standalone void-call statement (std::memcpy(...); etc.).
    pat2 = r"(?m)^[ \t]*((?:std::)?[A-Za-z_]\w*(?:::\w+)*)\((?:[^;{}()]|\([^()]*\))*\);[ \t]*\r?$"
    for m in re.finditer(pat2, text):
        if not is_code(m.start(), m.end()):
            continue
        head = m.group(1)
        if head in ("return", "if", "for", "while", "switch", "assert", "static_assert"):
            continue
        stmt = text[m.start() : m.end()].rstrip("\r")
        add(sites, "SDL", m.start(), stmt, " " * len(stmt), text)

    for s in sites:
        s["file"] = path
    return sites


def main():
    allsites = []
    for f in FILES:
        with open(f, "r", encoding="utf-8", newline="") as fh:
            text = fh.read()
        s = enumerate_sites(f, text)
        allsites.extend(s)
        print(f"{f}: {len(s)} sites", file=sys.stderr)
    print(f"TOTAL population: {len(allsites)} sites", file=sys.stderr)

    by_kind = {}
    for s in allsites:
        by_kind[s["kind"]] = by_kind.get(s["kind"], 0) + 1
    print("by kind: " + json.dumps(by_kind, sort_keys=True), file=sys.stderr)

    rng = random.Random(SEED)
    order = list(range(len(allsites)))
    rng.shuffle(order)
    out = {"seed": SEED, "files": FILES, "total": len(allsites), "by_kind": by_kind,
           "order": order, "sites": allsites}
    with open("experiments/popper2_population.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    print("wrote experiments/popper2_population.json", file=sys.stderr)


if __name__ == "__main__":
    main()

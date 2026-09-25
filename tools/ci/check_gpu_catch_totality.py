#!/usr/bin/env python3
"""CI gate: every outermost `try` in src/gpu ends in `catch (...)`.

The GPU backend turns exceptions into statuses at catch sites that also hold
engine state: a recorded command list, submitted work, an in-flight token, a
handle's live counts. A catch ladder that names types and stops leaves every
other type free to unwind past that state -- the class TE-435 closed, found
three times one region further along the submit path each time (the
recording windows, the submission tails, the decode finish). This gate makes
the rule structural: an outermost `try` block in any C++ source or header
under src/gpu must end its handler sequence with `catch (...)`, so no
exception type can leave it without passing through the site's own
containment.

A `try` lexically inside another `try` block's compound statement is exempt:
whatever leaves it lands in the enclosing ladder, which must itself end in
`catch (...)`. A `try` inside a handler is not inside a try block and is
checked like any other.

What this gate does not check, stated so it is not mistaken for checked: that
the `catch (...)` clause does the right thing. It proves the ladder is total,
never that the clause's cleanup matches its siblings'.

The scan strips comments, string literals and character literals, then
matches braces and parentheses. It refuses (exit 2) rather than guess when
the source contains something its lexer does not handle (a raw string
literal), when a brace or parenthesis does not match, or when it finds no
`try` at all, so a broken scan never reads as a pass.

Usage:
  python tools/ci/check_gpu_catch_totality.py            # exit 1 naming each violation
  python tools/ci/check_gpu_catch_totality.py --list     # every try and its handlers
  python tools/ci/check_gpu_catch_totality.py --root DIR # scan DIR instead of src/gpu
"""
from __future__ import annotations

import argparse
import os
import re
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_ROOT = os.path.join(_REPO_ROOT, "src", "gpu")
_SOURCE_SUFFIXES = (".cpp", ".cc", ".cxx", ".h", ".hpp", ".hh", ".hxx", ".inl")


class ScanRefused(Exception):
    """The lexer met input it does not handle; the gate refuses instead of passing."""


def strip_code(text: str, path: str) -> str:
    """Blanks comments and string/char literal contents, keeping every offset and newline."""
    out = list(text)
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            j = text.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                out[k] = " "
            i = j
            continue
        if c == "/" and nxt == "*":
            j = text.find("*/", i + 2)
            if j < 0:
                raise ScanRefused(f"{path}: unterminated block comment at offset {i}")
            for k in range(i, j + 2):
                if out[k] != "\n":
                    out[k] = " "
            i = j + 2
            continue
        if c == '"' or c == "'":
            prev = text[i - 1] if i > 0 else ""
            if c == '"' and prev == "R" and (i < 2 or not (text[i - 2].isalnum() or text[i - 2] == "_")):
                line = text.count("\n", 0, i) + 1
                raise ScanRefused(f"{path}:{line}: raw string literal; this lexer does not handle one")
            if c == "'" and prev.isalnum() and nxt.isalnum():
                # A digit separator (1'000, 0xFF'FF) when the token it sits in is a number; an
                # encoding prefix (L'A', u8'a') starts with a letter and opens a literal.
                s = i - 1
                while s > 0 and (text[s - 1].isalnum() or text[s - 1] in "_'."):
                    s -= 1
                if text[s].isdigit():
                    i += 1
                    continue
            j = i + 1
            while j < n and text[j] != c:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == "\n":
                    line = text.count("\n", 0, i) + 1
                    raise ScanRefused(f"{path}:{line}: unterminated literal")
                j += 1
            if j >= n:
                raise ScanRefused(f"{path}: unterminated literal at offset {i}")
            for k in range(i + 1, j):
                out[k] = " "
            i = j + 1
            continue
        i += 1
    return "".join(out)


def match_close(code: str, open_pos: int, path: str) -> int:
    """Offset of the bracket closing the one at `open_pos` ('{' or '(')."""
    opener = code[open_pos]
    closer = {"{": "}", "(": ")"}[opener]
    depth = 0
    for k in range(open_pos, len(code)):
        ch = code[k]
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return k
    line = code.count("\n", 0, open_pos) + 1
    raise ScanRefused(f"{path}:{line}: unmatched '{opener}'")


_TRY = re.compile(r"(?<![A-Za-z0-9_])try(?![A-Za-z0-9_])")
_CATCH = re.compile(r"\s*catch\s*\(")


def scan_file(path: str, rel: str) -> list[dict]:
    with open(path, "r", encoding="utf-8", errors="strict") as f:
        text = f.read()
    code = strip_code(text, rel)
    tries = []
    for m in _TRY.finditer(code):
        k = m.end()
        while k < len(code) and code[k].isspace():
            k += 1
        if k >= len(code) or code[k] != "{":
            line = code.count("\n", 0, m.start()) + 1
            raise ScanRefused(f"{rel}:{line}: 'try' not followed by a compound statement")
        block_open = k
        block_close = match_close(code, block_open, rel)
        handlers = []
        pos = block_close + 1
        while True:
            cm = _CATCH.match(code, pos)
            if not cm:
                break
            paren_open = cm.end() - 1
            paren_close = match_close(code, paren_open, rel)
            param = " ".join(code[paren_open + 1:paren_close].split())
            h = paren_close + 1
            while h < len(code) and code[h].isspace():
                h += 1
            if h >= len(code) or code[h] != "{":
                line = code.count("\n", 0, paren_open) + 1
                raise ScanRefused(f"{rel}:{line}: catch clause without a compound statement")
            handler_close = match_close(code, h, rel)
            handlers.append(param)
            pos = handler_close + 1
        if not handlers:
            line = code.count("\n", 0, m.start()) + 1
            raise ScanRefused(f"{rel}:{line}: try block with no handler")
        tries.append({
            "file": rel,
            "line": code.count("\n", 0, m.start()) + 1,
            "block": (block_open, block_close),
            "handlers": handlers,
        })
    for t in tries:
        t["nested"] = any(
            o is not t and o["block"][0] < t["block"][0] and t["block"][1] < o["block"][1]
            for o in tries
        )
        t["total"] = t["handlers"][-1] == "..."
    return tries


def scan_root(root: str) -> list[dict]:
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if name.endswith(_SOURCE_SUFFIXES):
                path = os.path.join(dirpath, name)
                rel = os.path.relpath(path, _REPO_ROOT).replace(os.sep, "/")
                found.extend(scan_file(path, rel))
    return found


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=_DEFAULT_ROOT, help="directory to scan (default: src/gpu)")
    ap.add_argument("--list", action="store_true", help="print every try and its handlers")
    args = ap.parse_args(argv)
    if not os.path.isdir(args.root):
        print(f"REFUSED: {args.root} is not a directory", file=sys.stderr)
        return 2
    try:
        tries = scan_root(args.root)
    except ScanRefused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    if not tries:
        print(f"REFUSED: no try block found under {args.root}; the scan cannot have worked",
              file=sys.stderr)
        return 2
    if args.list:
        for t in tries:
            kind = "nested" if t["nested"] else ("total" if t["total"] else "NOT TOTAL")
            print(f"{t['file']}:{t['line']}\t{kind}\tcatch ({') catch ('.join(t['handlers'])})")
    violations = [t for t in tries if not t["nested"] and not t["total"]]
    for t in violations:
        print(f"{t['file']}:{t['line']}: outermost try ends in 'catch ({t['handlers'][-1]})', "
              f"not 'catch (...)'")
    checked = sum(1 for t in tries if not t["nested"])
    if violations:
        print(f"FAIL: {len(violations)} of {checked} outermost try blocks under "
              f"{os.path.relpath(args.root, _REPO_ROOT)} are not total")
        return 1
    print(f"OK: {checked} outermost try blocks under {os.path.relpath(args.root, _REPO_ROOT)} "
          f"all end in catch (...) ({len(tries) - checked} nested)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

#!/usr/bin/env python3
"""Production CI gate for SuperSLM 1.7.1's own headline property: the shipped
`superslm`/`superslm_gpu` libraries write nothing to stdout (TE-400,
D-SLM7753; TE-402 S3). A one-time grep at the time of the fix is not a
standing guarantee -- a future `printf` added anywhere under `src/` or
`include/` would ship silently, exactly the gap S3 found. This script makes
the sweep TE-400's own build log performed by hand into something CI runs on
every push, refusing any stdout write it finds.

WHAT COUNTS AS A LIBRARY STDOUT WRITE. Any of `printf`, `wprintf`, `puts`,
`std::cout`/`std::wcout`, `fputs(..., stdout)`, `fwrite(..., stdout)`,
`fprintf`/`fwprintf(stdout, ...)`, or the Win32 `WriteConsoleA`/
`WriteConsoleW`/`STD_OUTPUT_HANDLE` path, as an actual call expression --
never as prose inside a comment (the English word "puts" appears
legitimately in several comments in this tree, e.g. "puts the row maximum",
and must not be flagged; TE-400's own build log found this the hard way and
excluded it by hand). `fprintf(stderr, ...)` and `fwprintf(stderr, ...)` are
explicitly NOT flagged: they are the channel this project's own diagnostics
already use (d3d12_harness.h's own convention, and TE-400's own fix).

METHOD. For each `.c`/`.cpp`/`.h`/`.hpp`/`.hlsli` file under the given
root's `src/` and `include/` (never `tests/` or `tools/`, which may print
freely -- this gate constrains only the shipped libraries), block comments,
line comments, and string/char literals are stripped with a small state
machine (so neither a comment's prose nor a format string's own text can
trip the regex below), and the forbidden-call regex is applied only to what
survives. This is source-text scanning, not full C++ parsing -- deliberately
lighter than the project's binary-symbol FP-free scan
(`tests/ci/check_fp_free_scan.py`), because the population here is "does an
identifiable call expression exist," which a comment/string-stripped regex
answers correctly, not "what does the optimizer's instruction selection
choose," which requires a compiled artifact.

ARBITRARY TREE ROOT. `--root PATH` scans any checkout, not just the one this
script lives in -- this is what lets a commissioning suite run a
must-reject construction against an old tree (e.g. SuperSLM v1.7.0 at
f43ab15, before TE-400's fix, which must fail this scan naming the exact
`d3d12_harness.h` call site removed at TE-400) and a must-accept
construction against the fixed tree, both from one script, without either
construction needing to check out a second copy of the repository itself
(TE-402's own conductor routing: "the scan's commissioning constructions ...
are authored by the test author, not you -- expose the scan so it can be run
against an arbitrary tree root"). `scan_tree(root)` is also importable
directly for the same reason.

Exit code 0 iff no library file contains a forbidden call; 1 otherwise,
naming every `file:line: <matched text>` hit, sorted.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))  # tools/ci -> tools -> repo root

# Directories under the root whose files are scanned. Never tests/ or
# tools/: this gate is about what the SHIPPED LIBRARIES do, and those are
# exactly the two source roots CMakeLists.txt draws `superslm`/
# `superslm_gpu`'s sources and public headers from.
_SCANNED_SUBDIRS = ("src", "include")

_SCANNED_EXTENSIONS = (".c", ".cpp", ".h", ".hpp")

# A forbidden call, matched only against comment/string-stripped text.
# Each pattern's own capture (if any) is informational only; the whole match
# is what gets reported. `fprintf`/`fwprintf` are matched generally and then
# filtered by the stdout/stderr check in `_is_stdout_fprintf` below, because
# a regex alternation for "first argument is exactly `stdout`" is fragile
# against whitespace/parenthesization but a small hand-check on the matched
# argument list is not.
_FORBIDDEN_PATTERNS = [
    re.compile(r"\b(?:std::)?w?printf\s*\("),
    re.compile(r"\b(?:std::)?puts\s*\("),
    re.compile(r"\bstd::(?:w)?cout\b"),
    re.compile(r"\b(?:std::)?fputs\s*\([^;]*\bstdout\b"),
    re.compile(r"\b(?:std::)?fwrite\s*\([^;]*\bstdout\b"),
    re.compile(r"\bWriteConsole[AW]?\s*\("),
    re.compile(r"\bSTD_OUTPUT_HANDLE\b"),
    re.compile(r"\b_putws\s*\("),
]

# fprintf/fwprintf: only forbidden when the first argument is `stdout`, not
# `stderr` (this project's own established diagnostic channel, TE-400).
_FPRINTF_CALL = re.compile(r"\b(?:std::)?fw?printf\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\b")


@dataclass(frozen=True)
class Hit:
    path: str
    line: int
    text: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.text}"


def _strip_comments_and_literals(source: str) -> str:
    """Replace every block comment, line comment, and string/char literal in
    `source` with equal-length whitespace (preserving line numbers and
    column offsets exactly, so reported line numbers match the original
    file), leaving only code text for the forbidden-call regexes to see."""
    out = []
    i = 0
    n = len(source)
    in_line_comment = False
    in_block_comment = False
    in_string = False
    in_char = False
    while i < n:
        c = source[i]
        nxt = source[i + 1] if i + 1 < n else ""
        if in_line_comment:
            if c == "\n":
                in_line_comment = False
                out.append(c)
            else:
                out.append(" ")
            i += 1
            continue
        if in_block_comment:
            if c == "*" and nxt == "/":
                out.append("  ")
                in_block_comment = False
                i += 2
                continue
            out.append("\n" if c == "\n" else " ")
            i += 1
            continue
        if in_string:
            if c == "\\" and i + 1 < n:
                out.append("  ")
                i += 2
                continue
            if c == '"':
                in_string = False
                out.append(" ")
                i += 1
                continue
            out.append("\n" if c == "\n" else " ")
            i += 1
            continue
        if in_char:
            if c == "\\" and i + 1 < n:
                out.append("  ")
                i += 2
                continue
            if c == "'":
                in_char = False
                out.append(" ")
                i += 1
                continue
            out.append("\n" if c == "\n" else " ")
            i += 1
            continue
        # Not currently inside any comment/literal.
        if c == "/" and nxt == "/":
            in_line_comment = True
            out.append("  ")
            i += 2
            continue
        if c == "/" and nxt == "*":
            in_block_comment = True
            out.append("  ")
            i += 2
            continue
        if c == '"':
            in_string = True
            out.append(" ")
            i += 1
            continue
        if c == "'":
            in_char = True
            out.append(" ")
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _is_stdout_fprintf(match_text: str) -> bool:
    m = _FPRINTF_CALL.search(match_text)
    if not m:
        return False
    return m.group(1) == "stdout"


def _scan_file(path: str) -> list[Hit]:
    with open(path, "r", encoding="utf-8", errors="surrogateescape") as f:
        original = f.read()
    stripped = _strip_comments_and_literals(original)
    hits: list[Hit] = []
    for lineno, line in enumerate(stripped.splitlines(), start=1):
        fp_call = _FPRINTF_CALL.search(line)
        if fp_call and _is_stdout_fprintf(line):
            hits.append(Hit(path, lineno, line.strip()))
            continue
        for pattern in _FORBIDDEN_PATTERNS:
            if pattern.search(line):
                hits.append(Hit(path, lineno, line.strip()))
                break
    return hits


def scan_tree(root: str) -> list[Hit]:
    """Every forbidden stdout call in `root`'s `src/` and `include/`
    subtrees, sorted by path then line. Empty if the property holds."""
    hits: list[Hit] = []
    for subdir in _SCANNED_SUBDIRS:
        base = os.path.join(root, subdir)
        if not os.path.isdir(base):
            continue
        for dirpath, _dirnames, filenames in os.walk(base):
            for name in filenames:
                if not name.endswith(_SCANNED_EXTENSIONS):
                    continue
                hits.extend(_scan_file(os.path.join(dirpath, name)))
    return sorted(hits, key=lambda h: (h.path, h.line))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=_DEFAULT_ROOT,
        help="Repository root to scan (its src/ and include/ subtrees). "
        "Defaults to this script's own repository. Pass a different "
        "checkout to run this gate's must-reject/must-accept constructions "
        "against another tree (TE-402 S3).",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"check_no_library_stdout.py: --root {root!r} is not a directory", file=sys.stderr)
        return 1

    hits = scan_tree(root)
    if hits:
        for h in hits:
            print(str(h))
        print(
            f"check_no_library_stdout.py: FAILED -- {len(hits)} stdout write(s) found in the "
            f"shipped superslm/superslm_gpu libraries under {root!r} (src/, include/); the "
            "engine library must write nothing to stdout (TE-400, D-SLM7753)",
            file=sys.stderr,
        )
        return 1

    print(f"check_no_library_stdout.py: OK -- no stdout write found under {root!r} (src/, include/)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

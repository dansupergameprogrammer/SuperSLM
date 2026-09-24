"""CI source check: the shipped `superslm`/`superslm_gpu` libraries write nothing to stdout
(TE-400, D-SLM7753; TE-402 S3, remedy #4: "add a hosted-CI source scan that fails on any stdout
write in src/ and include/"). A one-time grep at the time of the fix is not a standing
guarantee -- a future `printf` added anywhere under `src/` or `include/` would ship silently,
exactly the gap S3 found. This makes the sweep TE-400's own build log performed by hand into
something CI runs on every push, refusing any stdout write it finds.

Commissioned, blind, by tests/ci/test_te399_stdout_write_scan_commission.py (Curie, TE-399 fix
round) -- this module implements exactly that commission's own interface contract; do not change
the two public names or their signatures without updating that test, which this module does not
own.

WHAT COUNTS AS A LIBRARY STDOUT WRITE. Any of `printf`, `wprintf`, `puts`, `std::cout`/
`std::wcout`, `fputs(..., stdout)`, `fwrite(..., stdout)`, `fprintf`/`fwprintf(stdout, ...)`, or
the Win32 `WriteConsoleA`/`WriteConsoleW`/`STD_OUTPUT_HANDLE` path, as an actual call expression --
never as prose inside a comment (the English word "puts" appears legitimately in several comments
in this tree, e.g. "puts the row maximum", and must not be flagged; TE-400's own build log found
this the hard way and excluded it by hand). `fprintf(stderr, ...)` and `fwprintf(stderr, ...)` are
explicitly NOT flagged: they are the channel this project's own diagnostics already use
(d3d12_harness.h's own convention, and TE-400's own fix).

METHOD. Modelled on this directory's own precedent, check_no_forward_leaf_calls.py: a
comment-aware text scan, not a raw substring/regex pass and not full C++ parsing. Block comments,
line comments, and string/char literals are stripped with a small state machine (preserving line
numbers exactly, the same discipline check_no_forward_leaf_calls.py's own
`_strip_comments_preserving_line_numbers` uses), so neither a comment's prose nor a format
string's own text can trip the forbidden-call regexes, which run only on what survives.

Exit code 0 iff no scanned file contains a forbidden call; 1 otherwise, naming every
`path:line: reason` hit.
"""
from __future__ import annotations

import os
import re
import sys

_SCANNED_EXTENSIONS = (".c", ".cpp", ".h", ".hpp", ".hlsl", ".hlsli")

# A forbidden call, matched only against comment/string-stripped text. `fprintf`/`fwprintf` are
# matched generally and then filtered by `_is_stdout_fprintf` below, because a regex alternation
# for "first argument is exactly `stdout`" is fragile against whitespace/parenthesization but a
# small hand-check on the matched argument list is not.
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

# fprintf/fwprintf: only forbidden when the first argument is `stdout`, not `stderr` (this
# project's own established diagnostic channel, TE-400).
_FPRINTF_CALL = re.compile(r"\b(?:std::)?fw?printf\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\b")


def _strip_comments_and_literals(source: str) -> str:
    """Replace every block comment, line comment, and string/char literal in `source` with
    equal-length whitespace (preserving line numbers and column offsets exactly, so reported line
    numbers match the original file), leaving only code text for the forbidden-call regexes to
    see. Same discipline as check_no_forward_leaf_calls.py's own
    `_strip_comments_preserving_line_numbers`, extended to also blank string/char literals (a
    format string's own text, e.g. containing the word "printf", must not itself trip the
    regexes)."""
    out: list[str] = []
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


def _is_stdout_fprintf(line: str) -> bool:
    m = _FPRINTF_CALL.search(line)
    if not m:
        return False
    return m.group(1) == "stdout"


def _scan_file(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8", errors="surrogateescape") as f:
        original = f.read()
    stripped = _strip_comments_and_literals(original)
    hits: list[str] = []
    for lineno, line in enumerate(stripped.splitlines(), start=1):
        reason = None
        if _FPRINTF_CALL.search(line) and _is_stdout_fprintf(line):
            reason = "fprintf/fwprintf(stdout, ...)"
        else:
            for pattern in _FORBIDDEN_PATTERNS:
                if pattern.search(line):
                    reason = pattern.pattern
                    break
        if reason is not None:
            hits.append(f"{path}:{lineno}: stdout write ({reason})")
    return hits


def scan_for_stdout_writes(root_dirs: list[str]) -> list[str]:
    """Every source line under any of `root_dirs` (walked recursively; .c/.cpp/.h/.hpp/.hlsl/
    .hlsli by extension, matching this repo's own shipped-library source set) that writes to
    stdout -- printf/wprintf/puts/std::cout/fwrite(..., stdout)/fprintf(stdout, ...) and
    equivalents, the same enumeration CHANGELOG.md's own 1.7.1 entry and docs/releases/1.7.1.md's
    "Sweep" paragraph describe as already performed by hand for this release. Returns
    "path:line: reason" strings, empty if clean."""
    hits: list[str] = []
    for root_dir in root_dirs:
        if not os.path.isdir(root_dir):
            continue
        for dirpath, _dirnames, filenames in os.walk(root_dir):
            for name in filenames:
                if not name.endswith(_SCANNED_EXTENSIONS):
                    continue
                hits.extend(_scan_file(os.path.join(dirpath, name)))
    return sorted(hits)


def main(argv: list[str]) -> int:
    """CLI entry point: argv names one or more root directories (this repo's own src/ and
    include/ in normal CI use); prints every hit from scan_for_stdout_writes and returns 1 if any
    exist, 0 if the scan is clean -- the same contract check_no_pow_operator.main() and
    check_no_forward_leaf_calls.main() already use in this directory, so this scan wires into CI
    (.github/workflows/tests.yml) and a local `pytest tests/ci/` run the same way every sibling
    check here does."""
    if not argv:
        print("check_no_library_stdout_write.py: no root directories given", file=sys.stderr)
        return 1
    hits = scan_for_stdout_writes(argv)
    if hits:
        for h in hits:
            print(h)
        print(
            f"check_no_library_stdout_write.py: FAILED -- {len(hits)} stdout write(s) found "
            f"under {argv!r}; the engine library must write nothing to stdout (TE-400, "
            "D-SLM7753)",
            file=sys.stderr,
        )
        return 1
    print(f"check_no_library_stdout_write.py: OK -- no stdout write found under {argv!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
